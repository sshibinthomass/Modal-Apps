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
const ENDPOINT_URL = "https://sshibinthomass--flux-schnell-fluxmodel-generate-api.modal.run";

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

async function testInference(prompt: string): Promise<Buffer> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (MODAL_KEY) headers["Modal-Key"] = MODAL_KEY;
  if (MODAL_SECRET) headers["Modal-Secret"] = MODAL_SECRET;

  console.log(`Requesting image for prompt: '${prompt}'...`);
  const response = await fetch(ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({
      prompt,
      num_inference_steps: 4,
      guidance_scale: 0.0,
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
    const prompt = "A highly detailed close-up shot of a cybernetic eagle, glowing neon lines, digital art.";
    const imgBuffer = await testInference(prompt);

    // Ensure outputs directory exists
    const outputsDir = path.join(process.cwd(), "outputs");
    if (!fs.existsSync(outputsDir)) {
      fs.mkdirSync(outputsDir, { recursive: true });
    }

    const timestamp = getTimestamp();
    const filename = `flux_schnell_${timestamp}.png`;
    const outputPath = path.join(outputsDir, filename);

    fs.writeFileSync(outputPath, imgBuffer);
    console.log(`\nSuccess! Generated image saved to outputs/${filename} (${imgBuffer.length} bytes)`);
  } catch (err) {
    console.error("\nInference failed. Check logs above.");
    throw err;
  }
}

main().catch((err) => {
  console.error("Fatal error in main execution:", err.message || err);
  process.exit(1);
});
