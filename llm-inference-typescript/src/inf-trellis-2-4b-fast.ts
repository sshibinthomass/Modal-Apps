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
// Hashed/shortened because the full domain exceeds DNS character limit (63 chars)
const ENDPOINT_URL = "https://sshibinthomass--trellis-2-4b-fast-trellis2fastmodel-gene-1cf4f3.modal.run";

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

async function testInference(imgPath: string): Promise<Buffer> {
  const imgBuffer = fs.readFileSync(imgPath);
  const imageBase64 = imgBuffer.toString("base64");

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (MODAL_KEY) headers["Modal-Key"] = MODAL_KEY;
  if (MODAL_SECRET) headers["Modal-Secret"] = MODAL_SECRET;

  console.log("Sending request to Trellis 2-4B Fast secure endpoint...");
  const response = await fetch(ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image_base64: imageBase64,
      seed: 42,
      pipeline_type: "512",
      decimation_target: 100000,
      texture_size: 1024,
      remesh: false,
      webp: false,
      use_bf16: false,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function main() {
  console.log(`Target Endpoint URL: ${ENDPOINT_URL}\n`);
  try {
    // Dynamically locate img.png relative to running directory
    let imgPath = path.join(process.cwd(), "../llm-inference/img.png");
    if (!fs.existsSync(imgPath)) {
      imgPath = path.join(process.cwd(), "llm-inference/img.png");
    }

    if (!fs.existsSync(imgPath)) {
      throw new Error(`Could not find source image at ${imgPath}`);
    }

    const glbBuffer = await testInference(imgPath);

    // Ensure outputs directory exists
    const outputsDir = path.join(process.cwd(), "outputs");
    if (!fs.existsSync(outputsDir)) {
      fs.mkdirSync(outputsDir, { recursive: true });
    }

    const inputStem = path.basename(imgPath, path.extname(imgPath));
    const timestamp = getTimestamp();
    const filename = `${inputStem}_trellis_fast_${timestamp}.glb`;
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
