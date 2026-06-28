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
const ENDPOINT_URL = "https://sshibinthomass--rmbg-2-0-rmbgmodel-generate-api.modal.run";
const EXTRACT_ENDPOINT_URL = "https://sshibinthomass--rmbg-2-0-rmbgmodel-extract-and-remove-ba-111a3b.modal.run";

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

  console.log("Sending request to RMBG 2.0 secure endpoint (standard)...");
  const response = await fetch(ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image_base64: imageBase64
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function testExtractInference(imgPath: string, targetObject?: string): Promise<Buffer> {
  const imgBuffer = fs.readFileSync(imgPath);
  const imageBase64 = imgBuffer.toString("base64");

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (MODAL_KEY) headers["Modal-Key"] = MODAL_KEY;
  if (MODAL_SECRET) headers["Modal-Secret"] = MODAL_SECRET;

  console.log(`Sending request to RMBG 2.0 secure extract endpoint (target: ${targetObject || "auto-detect"})...`);
  const response = await fetch(EXTRACT_ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      image_base64: imageBase64,
      target_object: targetObject || null
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function main() {
  console.log(`Generate Endpoint URL: ${ENDPOINT_URL}`);
  console.log(`Extract Endpoint URL:  ${EXTRACT_ENDPOINT_URL}\n`);
  try {
    // Dynamically locate img.png relative to running directory
    let imgPath = path.join(process.cwd(), "../llm-inference/img.png");
    if (!fs.existsSync(imgPath)) {
      imgPath = path.join(process.cwd(), "llm-inference/img.png");
    }

    if (!fs.existsSync(imgPath)) {
      throw new Error(`Could not find source image at ${imgPath}`);
    }

    // Ensure outputs directory exists
    const outputsDir = path.join(process.cwd(), "outputs");
    if (!fs.existsSync(outputsDir)) {
      fs.mkdirSync(outputsDir, { recursive: true });
    }

    const inputStem = path.basename(imgPath, path.extname(imgPath));
    const timestamp = getTimestamp();

    // 1. Run standard inference
    const outputBuffer = await testInference(imgPath);
    const standardFilename = `${inputStem}_rmbg_${timestamp}.png`;
    const standardOutputPath = path.join(outputsDir, standardFilename);
    fs.writeFileSync(standardOutputPath, outputBuffer);
    console.log(`Success! Standard transparent image saved to outputs/${standardFilename} (${outputBuffer.length} bytes)`);

    // 2. Run extract inference (auto-detect)
    const extractBuffer = await testExtractInference(imgPath);
    const extractFilename = `${inputStem}_extracted_${timestamp}.png`;
    const extractOutputPath = path.join(outputsDir, extractFilename);
    fs.writeFileSync(extractOutputPath, extractBuffer);
    console.log(`Success! Extracted transparent image saved to outputs/${extractFilename} (${extractBuffer.length} bytes)`);

  } catch (err: any) {
    console.error("\nInference failed. Check logs above.");
    throw err;
  }
}

main().catch((err) => {
  console.error("Fatal error in main execution:", err.message || err);
  process.exit(1);
});
