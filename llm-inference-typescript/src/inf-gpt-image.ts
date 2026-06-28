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

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

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

/**
 * Checks if the image is a PNG and if its dimensions are divisible by 16.
 * OpenAI Images Edit API requires the image to be a PNG, less than 4MB, and have dimensions divisible by 16.
 */
function verifyPngDimensions(buffer: Buffer): { valid: boolean; width: number; height: number; reason?: string } {
  if (
    buffer.length < 24 ||
    buffer[0] !== 0x89 ||
    buffer[1] !== 0x50 ||
    buffer[2] !== 0x4e ||
    buffer[3] !== 0x47 ||
    buffer[4] !== 0x0d ||
    buffer[5] !== 0x0a ||
    buffer[6] !== 0x1a ||
    buffer[7] !== 0x0a
  ) {
    return { valid: false, width: 0, height: 0, reason: "Image is not a valid PNG file." };
  }

  // Check for IHDR chunk type
  const chunkType = buffer.toString("ascii", 12, 16);
  if (chunkType !== "IHDR") {
    return { valid: false, width: 0, height: 0, reason: "IHDR chunk not found in PNG file." };
  }

  const width = buffer.readInt32BE(16);
  const height = buffer.readInt32BE(20);

  if (width % 16 !== 0 || height % 16 !== 0) {
    return {
      valid: false,
      width,
      height,
      reason: `PNG dimensions (${width}x${height}) are not divisible by 16.`
    };
  }

  return { valid: true, width, height };
}

async function testGptImageInference(imgPath: string, targetObject?: string): Promise<Buffer> {
  if (!OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY environment variable is not defined.");
  }

  const imgBuffer = fs.readFileSync(imgPath);

  // Verify PNG compliance
  const verification = verifyPngDimensions(imgBuffer);
  if (!verification.valid) {
    console.warn(`[WARNING] Image preparation issue: ${verification.reason}`);
    console.warn("The OpenAI Images Edit API may fail if dimensions are not divisible by 16 or format is not PNG.");
  } else {
    console.log(`Image verified successfully: PNG format, dimensions ${verification.width}x${verification.height} (divisible by 16).`);
  }

  // Build the prompt as requested by the user
  let prompt = "";
  if (targetObject) {
    prompt = (
      `Extract the ${targetObject} from the image. ` +
      `Place the ${targetObject} in a frontal-side position suitable for 3D generation, ` +
      `and make the background solid pure white. The final output must contain only a single ${targetObject}, ` +
      `in high quality (HQ), extremely sharp, with clear details and studio lighting, optimized for 3D reconstruction.`
    );
    console.log(`Calling OpenAI gpt-image-2-2026-04-21 to extract '${targetObject}'...`);
  } else {
    prompt = (
      "Extract the main, most prominent object from the image. " +
      "Place it in a frontal-side position suitable for 3D generation, " +
      "and make the background solid pure white. The final output must contain only a single object, " +
      "in high quality (HQ), extremely sharp, with clear details and studio lighting, optimized for 3D reconstruction."
    );
    console.log("Calling OpenAI gpt-image-2-2026-04-21 to extract the main object...");
  }

  // Prepare FormData for the multipart POST request
  const formData = new FormData();
  const imageBlob = new Blob([imgBuffer], { type: "image/png" });
  formData.append("image", imageBlob, "input.png");
  formData.append("model", "gpt-image-2-2026-04-21");
  formData.append("prompt", prompt);
  formData.append("n", "1");
  try {
    const response = await fetch("https://api.openai.com/v1/images/edits", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`
      },
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = (await response.json()) as any;
    const imgData = data.data?.[0];

    if (!imgData) {
      throw new Error("No image data returned in OpenAI response.");
    }

    if (imgData.b64_json) {
      return Buffer.from(imgData.b64_json, "base64");
    } else if (imgData.url) {
      console.log(`Downloading generated image from URL: ${imgData.url}...`);
      const downloadRes = await fetch(imgData.url);
      if (!downloadRes.ok) {
        throw new Error(`Failed to download image from URL: ${downloadRes.statusText}`);
      }
      const arrayBuffer = await downloadRes.arrayBuffer();
      return Buffer.from(arrayBuffer);
    } else {
      throw new Error("Neither b64_json nor url was returned in OpenAI response.");
    }
  } catch (err: any) {
    console.error(`OpenAI API call failed: ${err.message || err}`);
    throw err;
  }
}

async function main() {
  try {
    // Locate img.png relative to running directory
    let imgPath = path.join(process.cwd(), "../llm-inference/img8.png");
    if (!fs.existsSync(imgPath)) {
      imgPath = path.join(process.cwd(), "llm-inference/img8.png");
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

    // Run inference with target object as an example (e.g. "chair") or auto-detect
    // You can modify targetObject or read it from arguments
    const targetObject = process.argv[2] || undefined;

    const outputBuffer = await testGptImageInference(imgPath, targetObject);

    const suffix = targetObject ? `_gpt_extracted_${targetObject}` : "_gpt_extracted";
    const filename = `${inputStem}${suffix}_${timestamp}.png`;
    const outputPath = path.join(outputsDir, filename);

    fs.writeFileSync(outputPath, outputBuffer);
    console.log(`\nSuccess! Generated image saved to outputs/${filename} (${outputBuffer.length} bytes)`);

  } catch (err: any) {
    console.error("\nInference failed. Check logs above.");
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error in main execution:", err.message || err);
  process.exit(1);
});
