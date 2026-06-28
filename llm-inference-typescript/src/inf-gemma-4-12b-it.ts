import * as dotenv from "dotenv";
import * as fs from "fs";

// Load environment variables by looking up to 2 directories up
for (const p of [".env", "../.env", "../../.env"]) {
  if (fs.existsSync(p)) {
    dotenv.config({ path: p });
    break;
  }
}

const { MODAL_KEY, MODAL_SECRET } = process.env;
const ENDPOINT_URL = "https://sshibinthomass--gemma-4-12b-it-gemmamodel-generate-api.modal.run";

async function testInference(prompt: string): Promise<string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (MODAL_KEY) headers["Modal-Key"] = MODAL_KEY;
  if (MODAL_SECRET) headers["Modal-Secret"] = MODAL_SECRET;

  console.log("Sending request to Gemma model...");
  const response = await fetch(ENDPOINT_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({ prompt, max_new_tokens: 256 }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function main() {
  console.log(`Target Endpoint URL: ${ENDPOINT_URL}\n`);
  const reply = await testInference("Explain what serverless computing is in three simple sentences.");
  console.log(`\nResponse:\n${reply}`);
}

main().catch((err) => console.error("\nExecution failed:", err.message || err));
