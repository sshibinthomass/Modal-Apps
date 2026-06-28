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
const ENDPOINT_URL = "https://sshibinthomass--image-to-3d-imageto3d-generate-api.modal.run";

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

  console.log("Sending request to Image-to-3D secure pipeline endpoint...");
  const response = await fetch(ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image_base64: imageBase64,
      seed: 42,
      pipeline_type: "1024_cascade",
      decimation_target: 1000000,
      texture_size: 4096,
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
    let imgPath = path.join(process.cwd(), "../llm-inference/img6.png");
    if (!fs.existsSync(imgPath)) {
      imgPath = path.join(process.cwd(), "llm-inference/img6.png");
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
    const filename = `${inputStem}_image_to_3d_${timestamp}.glb`;
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
