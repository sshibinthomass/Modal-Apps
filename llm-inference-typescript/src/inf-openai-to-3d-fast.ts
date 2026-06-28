import * as dotenv from "dotenv";
import * as fs from "fs";
import * as path from "path";

// Load environment variables by looking up to 2 directories up
for (const p of [".env", "../.env", "../../.env"]) {
  if (fs.existsSync(p)) {
    dotenv.config({ path: p });
    break;
  }
}

const { MODAL_KEY, MODAL_SECRET } = process.env;
const ENDPOINT_URL = "https://sshibinthomass--openai-to-3d-fast-openaiimageto3d-generate-api.modal.run";

function getTimestamp(): string {
  const now = new Date();
  const pad = (n: number) => n.toString().padStart(2, '0');
  const yyyy = now.getFullYear();
  const mm = pad(now.getMonth() + 1);
  const dd = pad(now.getDate());
  const hh = pad(now.getHours());
  const min = pad(now.getMinutes());
  const ss = pad(now.getSeconds());
  return `${yyyy}${mm}${dd}_${hh}${min}${ss}`;
}

async function testInference(imgPath: string, prompt?: string): Promise<Buffer> {
  const imgBuffer = fs.readFileSync(imgPath);
  const imageBase64 = imgBuffer.toString("base64");

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (MODAL_KEY) headers["Modal-Key"] = MODAL_KEY;
  if (MODAL_SECRET) headers["Modal-Secret"] = MODAL_SECRET;

  console.log(`Submitting OpenAI-to-3D job (prompt: '${prompt || "default"}')...`);
  
  // 1. Submit the job to start-api to get a call_id
  const startUrl = ENDPOINT_URL.replace("generate-api", "start-api");
  const startRes = await fetch(startUrl, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image_base64: imageBase64,
      prompt: prompt,
      seed: 42,
      pipeline_type: "512",
      decimation_target: 300000,
      texture_size: 1024,
    }),
  });

  if (!startRes.ok) {
    throw new Error(`Failed to start job: HTTP ${startRes.status}: ${await startRes.text()}`);
  }

  const { call_id } = (await startRes.json()) as { call_id: string };
  console.log(`Job submitted successfully! Call ID: ${call_id}. Polling for result...`);

  // 2. Poll result-api until completed
  const resultUrl = `${ENDPOINT_URL.replace("generate-api", "result-api")}?call_id=${call_id}`;
  while (true) {
    const resultRes = await fetch(resultUrl, {
      method: "GET",
      headers,
    });

    if (resultRes.status === 200) {
      console.log("\nSuccess! GLB file generated.");
      const arrayBuffer = await resultRes.arrayBuffer();
      return Buffer.from(arrayBuffer);
    } else if (resultRes.status === 202) {
      process.stdout.write(".");
      await new Promise((resolve) => setTimeout(resolve, 3000));
    } else {
      throw new Error(`\nPolling failed: HTTP ${resultRes.status}: ${await resultRes.text()}`);
    }
  }
}

async function main() {
  console.log(`Target Endpoint URL: ${ENDPOINT_URL}\n`);
  try {
    // Dynamically locate img8.png relative to running directory
    let imgPath = path.join(process.cwd(), "../llm-inference/img8.png");
    if (!fs.existsSync(imgPath)) {
      imgPath = path.join(process.cwd(), "llm-inference/img8.png");
    }

    if (!fs.existsSync(imgPath)) {
      throw new Error(`Could not find source image at ${imgPath}`);
    }

    const prompt = process.argv[2] || undefined;
    const glbBuffer = await testInference(imgPath, prompt);

    // Ensure outputs directory exists
    const outputsDir = path.join(process.cwd(), "outputs");
    if (!fs.existsSync(outputsDir)) {
      fs.mkdirSync(outputsDir, { recursive: true });
    }

    const inputStem = path.basename(imgPath, path.extname(imgPath));
    const timestamp = getTimestamp();
    const suffix = prompt ? `_openai_to_3d_${prompt}` : "_openai_to_3d";
    const filename = `${inputStem}${suffix}_${timestamp}.glb`;
    const outputPath = path.join(outputsDir, filename);

    fs.writeFileSync(outputPath, glbBuffer);
    console.log(`\nSuccess! Generated GLB saved to outputs/${filename} (${glbBuffer.length} bytes)`);
  } catch (err) {
    console.error("\nInference failed. Check logs above.");
    throw err;
  }
}

main().catch((err) => {
  console.error("Fatal error in main execution:", err.message || err);
  process.exit(1);
});
