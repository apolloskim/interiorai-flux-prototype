import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"
import sharp from 'sharp'
import OpenAI from 'openai'
import { analyzeImageWithGemini } from '@/app/utils/gemini'
import fs from 'fs'
import { Buffer } from 'buffer'
import * as fsPromises from 'fs/promises'
import path from 'path'

if (!process.env.FAL_KEY) {
  throw new Error("FAL_KEY environment variable is not set")
}

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY environment variable is not set")
}

// Optional: Make Flask backend URL configurable
const MASK_GENERATION_API_URL = process.env.MASK_GENERATION_API_URL || 'http://localhost:5001/generate_mask'
const VALIDATION_API_URL = process.env.VALIDATION_API_URL || 'http://localhost:5001/validate_layout'

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// --- Interfaces ---
interface ValidationError {
  object: string;
  check: 'Bounds' | 'SpatialAnchor' | 'Overlap' | 'DataError' | 'Setup';
  message: string;
  relatedObject?: string;
}

// Structure for items in structuredLayout
interface LayoutItem {
    object: string;
    description: string;
    prompt: string;
    bounding_box: [number, number, number, number];
    spatial_anchor: string;
}

// Structure for items in executionPlan
interface ExecutionStep {
    step: number;
    model: string;
    input_image: string; // Placeholder name
    object?: string; // Single object case
    objects?: string[]; // Multiple objects case
    prompt: string;
    mask: string; // Placeholder name - WILL BE REPLACED/USED
    controlnet_unions: any[]; // Define more specifically if possible
    note: string;
}


// NEW Interface for storing attempt results
interface AttemptResult {
  attemptNumber: number;
  structuredLayout: LayoutItem[] | null; // Use specific type
  executionPlan: ExecutionStep[] | null; // Use specific type
  validationStatus: string; // Keep as string to handle various statuses
  validationErrors: ValidationError[] | null;
  errorCount: number;
  analysisString: string | null; // Store the raw analysis string for this attempt
}

// Interface for the combined execution plan step (adding mask URL)
interface CombinedExecutionStep extends ExecutionStep {
    layoutDetails: LayoutItem[]; // Add the resolved layout info
    generatedMaskUrl?: string; // Store the URL of the generated mask for this step
}


async function getImageDimensions(buffer: Buffer): Promise<{ width: number; height: number }> {
  const metadata = await sharp(buffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Could not get image dimensions");
  }
  return { width: metadata.width, height: metadata.height };
}


async function resizeImage(imageUrl: string, targetWidth: number, targetHeight: number): Promise<Buffer> {
  const response = await fetch(imageUrl);
  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  return await sharp(buffer)
    .resize(targetWidth, targetHeight, {
      fit: 'fill'
    })
    .toBuffer();
}

async function generateCannyMap(imageUrl: string, originalWidth: number, originalHeight: number): Promise<string> {
  try {
    const result = await fal.subscribe("fal-ai/image-preprocessors/canny", {
      input: {
        image_url: imageUrl
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === "IN_PROGRESS") {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    // Resize the canny map
    const resizedCannyBuffer = await resizeImage((result.data as any).image.url, originalWidth, originalHeight);

    // Upload the resized canny map
    const resizedCannyFile = new File([resizedCannyBuffer], 'resized-canny.jpg', { type: 'image/jpeg' });
    const resizedCannyUrl = await fal.storage.upload(resizedCannyFile);

    return resizedCannyUrl;
  } catch (error) {
    console.error("Error generating canny map:", error);
    throw error;
  }
}

async function generateDepthMap(imageUrl: string, originalWidth: number, originalHeight: number): Promise<string> {
  try {
    const result = await fal.subscribe("fal-ai/imageutils/marigold-depth", {
      input: {
        image_url: imageUrl,
        ensemble_size: 20
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === "IN_PROGRESS") {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    // --- RETURN ORIGINAL URL DIRECTLY ---
    const originalDepthUrl = (result.data as any).image.url;
    if (typeof originalDepthUrl !== 'string' || !originalDepthUrl) {
      console.error("Marigold response structure:", JSON.stringify(result, null, 2));
      throw new Error("Depth map URL not found or not a string in Marigold response");
    }
    console.log("Using original Marigold depth map URL:", originalDepthUrl);
    return originalDepthUrl;
    // --- END RETURN ORIGINAL ---

  } catch (error) {
    console.error("Error generating depth map:", error);
    throw error;
  }
}

// Add the new function here
async function generateFurnishedImage(imageUrl: string, prompt: string): Promise<string> {
  try {
    console.log("Generating furnished image with prompt:", prompt);
    console.log("Using base image:", imageUrl);

    // Remove explicit type parameter and rely on inference/checking
    const result = await fal.subscribe("fal-ai/flux-general/image-to-image", {
      input: {
        // Dynamic values
        image_url: imageUrl,
        prompt: prompt, // Use the analysis from Gemini/GPT-4o here

        // Static parameters from user request
        loras: [],
        strength: 0.85,
        max_shift: 1.15,
        base_shift: 0.5,
        num_images: 1,
        controlnets: [],
        ip_adapters: [],
        control_loras: [],
        reference_end: 1,
        guidance_scale: 3,
        real_cfg_scale: 3.5,
        controlnet_unions: [],
        reference_strength: 0.65,
        num_inference_steps: 30,
        enable_safety_checker: true
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === "IN_PROGRESS") {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    // Check the result structure carefully using optional chaining
    const furnishedUrl = (result?.data as any)?.images?.[0]?.url;

    if (typeof furnishedUrl !== 'string' || !furnishedUrl) {
        console.error("Fal response structure:", JSON.stringify(result, null, 2));
        throw new Error("Furnished image URL not found or not a string in fal response");
    }

    console.log("Furnished image generated:", furnishedUrl);
    return furnishedUrl;
  } catch (error) {
    console.error("Error generating furnished image:", error);
    throw error;
  }
}

// Function to call SA2VA model
async function getFurnitureListFromSA2VA(imageUrl: string): Promise<string> {
  const prompt = "Please look thoroughly at the image and list every piece of furniture and decor visible. Include all items such as sofas, chairs, tables, rugs, potted plants, light fixtures, wall art, and any other furnishing or decorative object. Do not include structural elements like walls, windows, ceilings, or floors. If there are multiple similar items, distinguish them using consistent numbered labels (e.g., \"potted plant 1\", \"potted plant 2\", \"wall frame 1\", \"wall frame 2\"). Do not use spatial terms like \"left\", \"right\", or \"center\". Provide the result as a single, comma-separated list with no numbers or extra commentary—just the item names."
  console.log("Calling SA2VA with image:", imageUrl);
  try {
    const result = await fal.subscribe("fal-ai/sa2va/8b/image", {
      input: {
        prompt: prompt,
        image_url: imageUrl
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === "IN_PROGRESS") {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    // Correctly access the output field based on the logged structure
    const rawText = (result?.data as any)?.output;
    if (typeof rawText !== 'string' || !rawText) {
      console.error("SA2VA response structure:", JSON.stringify(result, null, 2));
      throw new Error("Furniture list not found or not a string in SA2VA response");
    }

    // Parse the output
    const parsedText = rawText.replace(/<\|im_end\|>$/, '').trim();
    console.log("SA2VA Parsed Furniture List:", parsedText);
    return parsedText;

  } catch (error) {
    console.error("Error calling SA2VA model:", error);
    throw error;
  }
}

// Function to call Florence-2 model for grounding
async function getGroundingDataFromFlorence(imageUrl: string, furnitureList: string): Promise<any> {
  console.log("Calling Florence-2 with image:", imageUrl, "and text:", furnitureList);
  try {
    const result = await fal.subscribe("fal-ai/florence-2-large/caption-to-phrase-grounding", {
      input: {
        image_url: imageUrl,
        text_input: furnitureList // Use the parsed list from SA2VA
      },
      logs: true,
      onQueueUpdate: (update) => {
        if (update.status === "IN_PROGRESS") {
          update.logs.map((log) => log.message).forEach(console.log);
        }
      },
    });

    // Return the entire data part of the result for grounding info
    const groundingData = result?.data;
    if (!groundingData) {
      console.error("Florence-2 response structure:", JSON.stringify(result, null, 2));
      throw new Error("Grounding data not found in Florence-2 response");
    }

    console.log("Florence-2 Grounding Data:", groundingData);
    return groundingData;

  } catch (error) {
    console.error("Error calling Florence-2 model:", error);
    throw error;
  }
}

// Helper Function to Prepare Gemini Prompt
function prepareGeminiPrompt(
    validationFeedback: string, // Pass errors as a formatted string or empty string
    florenceResults: any,
    userPrompt: string,
    imageWidth: number,
    imageHeight: number
): string {
    const promptTemplate = fs.readFileSync('src/app/utils/goated-gemini-prompt.txt', 'utf8');
    return promptTemplate
        .replace('[VALIDATION_FEEDBACK_SECTION]', validationFeedback) // Insert feedback
        .replace('[PLACEHOLDER FOR IMAGE_DIMENSIONS]', `${imageWidth}x${imageHeight}`)
        .replace('[PLACEHOLDER FOR USER_STYLE_PROMPT]', userPrompt)
        .replace('[PLACEHOLDER FOR FLORENCE_JSON]', JSON.stringify(florenceResults));
}

// Helper Function to Parse Gemini Response
function parseGeminiResponse(analysisString: string | null | undefined): { structuredLayout: LayoutItem[] | null, executionPlan: ExecutionStep[] | null } {
    let structuredLayout: LayoutItem[] | null = null;
    let executionPlan: ExecutionStep[] | null = null;

    if (typeof analysisString !== 'string' || !analysisString) {
        console.error("Analysis result is null or not a string:", analysisString);
        return { structuredLayout, executionPlan };
    }

    let jsonStringToParse = analysisString.trim(); // Trim whitespace first
    let didExtract = false;

    // Try extracting content if fences are present
    if (jsonStringToParse.startsWith("```json") && jsonStringToParse.endsWith("```")) {
        console.log("Detected JSON fences. Attempting to extract content between first '{' and last '}'.");
        const startIndex = jsonStringToParse.indexOf('{');
        const endIndex = jsonStringToParse.lastIndexOf('}');

        if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
            jsonStringToParse = jsonStringToParse.substring(startIndex, endIndex + 1);
            console.log("Successfully extracted substring between braces.");
            didExtract = true;
        } else {
            console.warn("Found fences but failed to find valid start/end braces '{}'. Will attempt to parse the trimmed string without fences.");
             // Fallback: try removing fences crudely if brace finding failed
             jsonStringToParse = jsonStringToParse.replace(/^```json\s*/, '').replace(/\s*```$/, '');
        }
    } else {
        console.log("No JSON fences detected. Attempting to parse the string directly.");
    }

    try {
        const analysisJson = JSON.parse(jsonStringToParse);
        // console.log("Successfully parsed JSON content."); // Keep log minimal on success

        if (analysisJson && typeof analysisJson === 'object') {
             if (Array.isArray(analysisJson.structured_layout)) {
                 structuredLayout = analysisJson.structured_layout;
                 // console.log("Successfully extracted structured_layout.");
             } else {
                  console.error("Parsed JSON does not contain a valid 'structured_layout' array.");
             }
             if (Array.isArray(analysisJson.execution_plan)) {
                 executionPlan = analysisJson.execution_plan;
                  // console.log("Successfully extracted execution_plan.");
             } else {
                  console.error("Parsed JSON does not contain a valid 'execution_plan' array.");
             }
           } else {
              console.error("Parsed content is not a valid object:", analysisJson);
           }

    } catch (parseError: unknown) {
        console.error("Failed to parse JSON string:", parseError);
        console.error(`String attempted to parse (extracted: ${didExtract}):`, jsonStringToParse); // Log the string that failed + if extraction happened

        // Type check before accessing properties
        const errorMessage = parseError instanceof Error ? parseError.message : String(parseError);

        // Re-throw error to be caught by the main handler
        throw new Error(`Failed to parse JSON response after processing: ${errorMessage}`);
    }

    return { structuredLayout, executionPlan };
}

// Helper Function to Validate Layout
async function validateLayoutWithApi(
    layout: LayoutItem[] | null,
    depthMapUrl: string | null,
    imageWidth: number,
    imageHeight: number
): Promise<{ validationStatus: string, validationErrors: ValidationError[] }> {
    let validationErrors: ValidationError[] = [];
    let validationStatus: string = "not_run";

    if (layout && depthMapUrl && imageWidth && imageHeight) {
      // const validationApiUrl = process.env.VALIDATION_API_URL || 'http://localhost:5001/validate_layout'; // Use constant
      console.log(`Calling validation API at: ${VALIDATION_API_URL}`);

      try {
        const validationResponse = await fetch(VALIDATION_API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            layout: layout,
            image_dimensions: { width: imageWidth, height: imageHeight },
            depth_map_url: depthMapUrl,
          }),
          signal: AbortSignal.timeout(30000)
        });

        if (!validationResponse.ok) {
          const errorBody = await validationResponse.text();
          console.error(`Validation API returned error ${validationResponse.status}: ${errorBody}`);
          validationStatus = "api_error";
        } else {
          const validationResult = await validationResponse.json();
          if (validationResult.status === 'success') {
            validationStatus = "success";
            console.log("Layout validation successful.");
          } else if (validationResult.status === 'error') {
            validationStatus = "failed";
            // Ensure validationErrors conforms to the interface
            validationErrors = (validationResult.errors || []).map((err: any): ValidationError => ({
                 object: typeof err.object === 'string' ? err.object : 'unknown',
                 check: typeof err.check === 'string' ? err.check : 'UnknownCheck', // Add default or handle error
                 message: typeof err.message === 'string' ? err.message : 'Unknown error message',
                 relatedObject: typeof err.relatedObject === 'string' ? err.relatedObject : undefined,
            }));
            console.warn(`Layout validation failed with ${validationErrors.length} errors.`);
            console.warn("Validation Errors:", JSON.stringify(validationErrors, null, 2));
          } else {
             validationStatus = "unknown_response";
             console.error("Validation API returned unexpected status:", validationResult);
          }
        }
      } catch (validationError) {
         validationStatus = "network_error";
         // ... (error logging for network/timeout) ...
         if (validationError instanceof Error && validationError.name === 'TimeoutError') {
             console.error("Validation API call timed out:", validationError);
         } else {
            console.error("Error calling validation API:", validationError);
         }
      }
    } else {
        console.warn("Skipping validation API call due to missing data (layout, depth map, or dimensions).");
        validationStatus = "skipped";
    }
    return { validationStatus, validationErrors };
}

// --- Add Helper Function ---
/**
 * Generates the validation feedback section for re-prompting the LLM.
 * Uses qualitative instructions for anchor errors.
 * @param validationErrors - The array of error objects from the validator API.
 * @param depthConventionHighIsClose - Set based on depth map (false for Marigold).
 * @returns The formatted feedback string section, or null if no errors.
 */
function generateRepromptFeedback(
    validationErrors: ValidationError[] | null | undefined,
    depthConventionHighIsClose: boolean = false // Default to Marigold (Low=Close)
): string | null {
  if (!validationErrors || validationErrors.length === 0) {
    return null; // No errors, no feedback needed
  }

  const feedback_instructions: string[] = [];
  const objects_to_revise: Set<string> = new Set();

  // Regex to try and extract depth values for context, even if not used in instruction
  const anchorRegex = /Object Img Depth \((\d+(\.\d+)?)\) outside Plane Z Range \[(-?\d+(\.\d+)?)-(\d+(\.\d+)?)\].*/;

  for (const error of validationErrors) {
    const obj_name = error.object;
    objects_to_revise.add(obj_name);
    if (error.relatedObject) {
      objects_to_revise.add(error.relatedObject);
    }

    let instruction: string | null = null;
    const check_type = error.check;
    const message = error.message || "";

    try {
      if (check_type === "SpatialAnchor") {
        const anchor = message.split("'")[1] || 'unknown'; // Extract anchor type simply
        const match = message.match(anchorRegex);
        let direction = "Incorrect depth placement"; // Default direction

        if (match) {
          const obj_depth = parseFloat(match[1]);
          const expected_min = parseFloat(match[3]);
          const expected_max = parseFloat(match[5]);

          // Determine direction based on specified convention
          if (depthConventionHighIsClose) { // High = Close
            if (obj_depth > expected_max) direction = "FARTHER (higher depth value)";
            else if (obj_depth < expected_min) direction = "CLOSER (lower depth value)";
          } else { // Low = Close (Marigold)
            if (obj_depth > expected_max) direction = "FARTHER (higher depth value)";
            else if (obj_depth < expected_min) direction = "CLOSER (lower depth value)";
          }

          // --- Generate Qualitative Instruction ---
          if (anchor === 'ceiling') {
              if (direction.includes("FARTHER")) {
                  instruction = `Place '${obj_name}' (anchor 'ceiling') **much higher (significantly smaller Y coordinates)**. It is currently placed too low/far from the main ceiling plane.`;
              } else { // Placed too close (unlikely for pendants, but possible)
                   instruction = `Place '${obj_name}' (anchor 'ceiling') **slightly lower (slightly larger Y coordinates)**. It is currently placed too high/close to the main ceiling plane.`;
              }
          } else if (anchor === 'wall') {
              if (direction.includes("FARTHER")) {
                   instruction = `Place '${obj_name}' (anchor 'wall') **significantly closer (lower depth value)**. It is currently placed too far back compared to the main wall plane it should be on. Check X/Y placement.`;
              } else { // Placed too close
                   instruction = `Place '${obj_name}' (anchor 'wall') **significantly farther back (higher depth value)**. It is currently placed too close compared to the main wall plane it should be on. Check X/Y placement.`;
              }
          } else if (anchor === 'floor') {
              if (direction.includes("FARTHER")) {
                   instruction = `Place '${obj_name}' (anchor 'floor') **closer (lower depth value)**. It is currently placed too far back compared to the expected floor plane depth in its area. Adjust Y coordinate.`;
              } else { // Placed too close
                   instruction = `Place '${obj_name}' (anchor 'floor') **farther back (higher depth value)**. It is currently placed too close compared to the expected floor plane depth in its area. Adjust Y coordinate.`;
              }
          } else {
               instruction = `Re-evaluate placement for '${obj_name}' (anchor '${anchor}'). Failed depth consistency check. Direction: ${direction}.`;
          }

        } else {
           // Fallback if regex fails
            const anchorName = error.object || 'unknown_object';
           instruction = `Revise placement depth for '${obj_name}' (anchor '${anchorName}') - failed depth consistency check. Message: ${message}`;
        }

      } else if (check_type === "Overlap") {
         const relatedObjName = error.relatedObject || 'another_object';
        instruction = `Resolve collision risk between '${obj_name}' and '${relatedObjName}'. Increase separation between their bounding boxes while maintaining realistic placement.`;
      } else if (check_type === "Bounds") {
         const dimsMatch = message.match(/\((\d+)x(\d+)\)/);
         const dims = dimsMatch ? `(${dimsMatch[1]}x${dimsMatch[2]})` : '';
         instruction = `Correct bounding box for '${obj_name}'; it extends outside image dimensions ${dims}. Ensure coordinates [xmin, ymin, xmax, ymax] are within bounds.`;
      } else if (check_type === "DataError") {
          instruction = `Correct data format for '${obj_name}'. Error: ${message}`;
      } else if (check_type === "Setup") {
           console.error(`Setup/System Error: ${message}`);
           instruction = null;
      } else {
           instruction = `Address issue for '${obj_name}'. Check: ${check_type}, Message: ${message}`;
      }
    } catch (parseError: unknown) {
         const errorMsg = parseError instanceof Error ? parseError.message : String(parseError);
        console.error(`Error parsing validation message for ${obj_name}: ${message}`, errorMsg);
        instruction = `Address validation failure for '${obj_name}'. Check: ${check_type}. Message: ${message}`;
    }

    if (instruction) {
      feedback_instructions.push(instruction);
    }
  }

  // --- Construct feedback section (Same as before) ---
  if (feedback_instructions.length === 0) { return null; }
  const feedbackSection = `
ISSUE: The previously generated plan failed geometric validation.

VALIDATION ERRORS:
\`\`\`json
${JSON.stringify(validationErrors, null, 2)}
\`\`\`

REQUEST: Please generate a **revised** \`structured_layout\` and \`execution_plan\` JSON. Specifically address the VALIDATION ERRORS by revising the objects: ${JSON.stringify(Array.from(objects_to_revise).sort())}. Follow these specific instructions derived from the errors:
${feedback_instructions.map(inst => `- ${inst}`).join('\n')}
Adhere to all original requirements and guardrails. Ensure the output is ONLY the raw JSON.
`;
  return feedbackSection;
}

// --- MAIN POST FUNCTION (Modify loop inside) ---
export async function POST(request: NextRequest) {
  let generatedFeedbacks: string[] = [];
  let originalImageUrl: string | null = null;
  let cannyMapUrl: string | null = null;
  let depthMapUrl: string | null = null;
  let furnishedImageUrl: string | null = null;
  let furnitureList: string | null = null;
  let groundingData: any = null;
  let florenceResults: any = null;
  let originalWidth: number = 0;
  let originalHeight: number = 0;
  let userPrompt: string = "";
  let model: 'gpt4o' | 'gemini' = 'gemini';
  let attemptResults: AttemptResult[] = [];
  const maxAttempts = 3;

  try {
    // --- Initial Setup ---
    const formData = await request.formData();
    const image = formData.get("image") as File;
    userPrompt = (formData.get("prompt") || formData.get("text")) as string;
    model = (formData.get("model") || 'gemini') as 'gpt4o' | 'gemini';

    if (!image || !userPrompt) {
      const error = !image ? "No image provided" : "No prompt provided";
      return NextResponse.json({ success: false, error }, { status: 400 });
    }
    console.log("Received prompt:", userPrompt);

    const imageBuffer = Buffer.from(await image.arrayBuffer());
    ({ width: originalWidth, height: originalHeight } = await getImageDimensions(imageBuffer));
    const imageDimensions = { width: originalWidth, height: originalHeight }; // Store dimensions
    console.log("Original image dimensions:", imageDimensions);
    originalImageUrl = await fal.storage.upload(image);

    // --- Generate Supporting Assets (Done once) ---
    console.log("\n--- Generating Supporting Assets ---");
    if (!originalImageUrl) throw new Error("Original image URL is null after upload."); // Add null check
    [cannyMapUrl, depthMapUrl] = await Promise.all([
      generateCannyMap(originalImageUrl!, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl!, originalWidth, originalHeight),
    ]);
    if (!cannyMapUrl || !depthMapUrl) throw new Error("Failed to generate Canny or Depth map URL."); // Add null checks
    furnishedImageUrl = await generateFurnishedImage(originalImageUrl!, userPrompt);
    furnitureList = await getFurnitureListFromSA2VA(furnishedImageUrl!);
    groundingData = await getGroundingDataFromFlorence(furnishedImageUrl!, furnitureList!);
    florenceResults = groundingData.results;
    console.log("--- Supporting Assets Generated ---");

    // --- Loop for Generation and Validation ---
    let currentValidationFeedback = "";
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`\n--- Attempt ${attempt} of ${maxAttempts}: Layout Generation ---`);
      let currentAnalysisString: string | null = null;
      let currentLayout: LayoutItem[] | null = null;
      let currentPlan: ExecutionStep[] | null = null;
      let currentValidationStatus: string = "not_run";
      let currentValidationErrors: ValidationError[] = [];

      let currentPrompt = prepareGeminiPrompt(
        currentValidationFeedback, florenceResults!, userPrompt, originalWidth, originalHeight
      );

      try {
        currentAnalysisString = await analyzeImageWithGemini(
          originalImageUrl!, depthMapUrl!, cannyMapUrl!, furnishedImageUrl!,
          florenceResults!, currentPrompt, originalWidth, originalHeight
        );
      } catch (geminiError: any) {
        console.error(`Attempt ${attempt}: Error calling Gemini - ${geminiError.message}`);
        currentValidationStatus = `attempt_${attempt}_gemini_error`;
        attemptResults.push({
            attemptNumber: attempt, structuredLayout: null, executionPlan: null,
            validationStatus: currentValidationStatus, validationErrors: [], errorCount: 999, analysisString: null,
        });
        if (attempt === maxAttempts) break;
        currentValidationFeedback = "Gemini failed to respond. Retrying generation.";
        generatedFeedbacks.push(currentValidationFeedback);
        continue;
      }

      try {
           ({ structuredLayout: currentLayout, executionPlan: currentPlan } = parseGeminiResponse(currentAnalysisString));
           if (!currentLayout || !currentPlan || currentLayout.length === 0 || currentPlan.length === 0) {
                throw new Error("Parsing failed or returned null/empty layout or plan.");
           }
      } catch (parseError: any) {
          console.error(`Attempt ${attempt}: Parsing failed - ${parseError.message}`);
          currentValidationStatus = `attempt_${attempt}_parsing_failed`;
           attemptResults.push({
                attemptNumber: attempt, structuredLayout: null, executionPlan: null,
                validationStatus: currentValidationStatus, validationErrors: [], errorCount: 998,
                analysisString: currentAnalysisString,
           });
          if (attempt === maxAttempts) break;
          currentValidationFeedback = "Failed to parse the previous response. Retrying generation with stricter JSON format request.";
          generatedFeedbacks.push(currentValidationFeedback);
          continue;
      }

      console.log(`--- Attempt ${attempt}: Validating Layout ---`);
      ({ validationStatus: currentValidationStatus, validationErrors: currentValidationErrors } = await validateLayoutWithApi(
        currentLayout, depthMapUrl!, originalWidth, originalHeight
      ));

      const currentErrorCount = currentValidationErrors?.length ?? 0;
      attemptResults.push({
        attemptNumber: attempt, structuredLayout: currentLayout, executionPlan: currentPlan,
        validationStatus: currentValidationStatus, validationErrors: currentValidationErrors,
        errorCount: currentErrorCount, analysisString: currentAnalysisString,
      });

      if (currentValidationStatus === 'success') {
        console.log(`Attempt ${attempt}: Validation successful!`);
        break;
      } else {
        console.warn(`Attempt ${attempt}: Validation status: ${currentValidationStatus}.`);
        if (attempt < maxAttempts) {
          console.log(`Generating intelligent feedback for attempt ${attempt + 1}...`);
          const generatedFeedback = generateRepromptFeedback(currentValidationErrors);
          if (generatedFeedback) {
            currentValidationFeedback = generatedFeedback;
            generatedFeedbacks.push(generatedFeedback);
          } else {
            console.warn(`Attempt ${attempt}: Validation failed (${currentValidationStatus}), but no actionable feedback generated. Stopping retries.`);
            currentValidationFeedback = "";
            break;
          }
        } else {
          console.error(`Max attempts (${maxAttempts}) reached. Final validation status: ${currentValidationStatus}`);
        }
      }
    } // End of attempt loop

    // --- Loop Finished - Determine Best Attempt ---
    console.log("\n--- Determining Best Attempt ---");
    let bestAttempt: AttemptResult | null = null;
    if (attemptResults.length > 0) {
      const successfulAttempts = attemptResults.filter(r => r.validationStatus === 'success');
      if (successfulAttempts.length > 0) {
        bestAttempt = successfulAttempts[0];
        console.log(`Selected best attempt: #${bestAttempt.attemptNumber} (Validation Passed)`);
      } else {
        const validAttempts = attemptResults.filter(r => r.errorCount < 900 && r.structuredLayout && r.executionPlan);
         if (validAttempts.length > 0) {
            validAttempts.sort((a, b) => a.errorCount - b.errorCount);
            bestAttempt = validAttempts[0];
             console.log(`Selected best attempt: #${bestAttempt.attemptNumber} (Failed Validation, Min Errors: ${bestAttempt.errorCount})`);
         } else {
             const attemptsWithData = attemptResults.filter(r => r.analysisString);
             if (attemptsWithData.length > 0) {
                bestAttempt = attemptsWithData[attemptsWithData.length - 1];
                 console.warn(`Selected best attempt: #${bestAttempt.attemptNumber} (Failed early: ${bestAttempt.validationStatus}). Layout/Plan might be missing or invalid.`);
             } else {
             bestAttempt = attemptResults[attemptResults.length - 1];
                 console.error(`Selected best attempt: #${bestAttempt?.attemptNumber}. Failed very early: ${bestAttempt?.validationStatus}. No layout or plan available.`);
             }
         }
      }
    } else {
      console.error("No attempts were completed or stored. Cannot proceed.");
       return NextResponse.json(
           { success: false, error: "Failed to generate or validate any layout/plan after multiple attempts." }, { status: 500 }
       );
    }

     // --- MAP EXECUTION PLAN TO LAYOUT DETAILS ---
     console.log("\n--- Mapping Execution Plan to Layout Details ---");
     let combinedExecutionPlan: CombinedExecutionStep[] = [];
     const finalStructuredLayout = bestAttempt?.structuredLayout ?? [];
     const finalExecutionPlan = bestAttempt?.executionPlan ?? [];

     if (finalStructuredLayout.length > 0 && finalExecutionPlan.length > 0) {
         const layoutMap = new Map<string, LayoutItem>();
         for (const layoutItem of finalStructuredLayout) {
             if (layoutItem && layoutItem.object) { layoutMap.set(layoutItem.object, layoutItem); }
             else { console.warn("Skipping invalid item during layout map creation:", layoutItem); }
         }
         console.log(`Created layout map with ${layoutMap.size} items.`);

         for (const step of finalExecutionPlan) {
             if (!step || typeof step.step !== 'number') {
                 console.warn("Skipping invalid step in final executionPlan:", step); continue;
             }
             const stepObjects: string[] = [];
             if (typeof step.object === 'string' && step.object) { stepObjects.push(step.object); }
             else if (Array.isArray(step.objects) && step.objects.length > 0) {
                 stepObjects.push(...step.objects.filter(obj => typeof obj === 'string' && obj));
             }
             if (stepObjects.length === 0) {
                  console.warn(`Step ${step.step}: No valid object names found. Skipping step.`); continue;
             }
             const layoutDetailsForStep: LayoutItem[] = [];
             let missingLayout = false;
             for (const objName of stepObjects) {
                 const layout = layoutMap.get(objName);
                 if (layout) { layoutDetailsForStep.push(layout); }
                 else {
                     console.error(`Step ${step.step}: Layout details not found for object '${objName}'. This step cannot be executed.`);
                     missingLayout = true; break;
                 }
             }
             if (!missingLayout && layoutDetailsForStep.length > 0) {
                 const combinedStep: CombinedExecutionStep = {
                     ...(step as ExecutionStep), layoutDetails: layoutDetailsForStep
                 };
                 combinedExecutionPlan.push(combinedStep);
             } else if (missingLayout) {
                 console.warn(`Skipping Step ${step.step} due to missing layout data.`);
             } else {
                  console.warn(`Step ${step.step}: No valid layout details could be matched. Skipping step.`);
             }
         }
         console.log(`Successfully mapped ${combinedExecutionPlan.length} execution steps.`);
     } else {
         console.error("Cannot create combined execution plan: Final layout or plan is missing/empty.");
         return NextResponse.json(
           {
                success: false,
                error: "Could not proceed: Missing valid layout or plan."
                // Removed problematic details spread syntax
           },
           { status: 500 }
       );
     }

     // --- START MASK GENERATION AND EXECUTION LOOP ---
     console.log("\n--- Starting Mask Generation & Execution Loop ---");
     let currentImageUrl = originalImageUrl; // Initialize with the empty room URL
     let finalGeneratedImageUrl: string | null = originalImageUrl; // Track final result

     // REMOVED: Local mask directory creation logic
     // const localMaskDir = path.join(process.cwd(), 'public', 'generated_masks');
     // try { ... } catch { ... }
     // -----------------------------------------

     for (let i = 0; i < combinedExecutionPlan.length; i++) {
         const combinedStep = combinedExecutionPlan[i];
         console.log(`\n--- Processing Step ${combinedStep.step} (${combinedStep.note || 'No note'}) ---`);

         const boundingBoxesForStep = combinedStep.layoutDetails.map(ld => ld.bounding_box);
         if (!boundingBoxesForStep || boundingBoxesForStep.length === 0) {
             console.warn(`Step ${combinedStep.step}: No bounding boxes found in layoutDetails. Skipping mask generation and execution.`);
             continue;
         }
         console.log(`Step ${combinedStep.step}: Found ${boundingBoxesForStep.length} bounding box(es).`);

         let maskUrlForStep: string | null = null;
         try {
             console.log(`Step ${combinedStep.step}: Calling Mask Generation API at ${MASK_GENERATION_API_URL}`);
             const maskPayload = {
                 bounding_boxes: boundingBoxesForStep,
                 image_dimensions: imageDimensions,
             };

             const maskResponse = await fetch(MASK_GENERATION_API_URL, {
                 method: 'POST',
                 headers: { 'Content-Type': 'application/json' },
                 body: JSON.stringify(maskPayload),
                 signal: AbortSignal.timeout(30000)
             });

             if (!maskResponse.ok) {
                 const errorBody = await maskResponse.text();
                 throw new Error(`Mask generation API failed with status ${maskResponse.status}: ${errorBody}`);
             }

             const maskResult = await maskResponse.json();
             if (maskResult.status !== 'success' || !maskResult.mask_b64) {
                 throw new Error(`Mask generation API did not return success or mask_b64: ${JSON.stringify(maskResult)}`);
             }

             console.log(`Step ${combinedStep.step}: Received mask base64, decoding...`);
             const maskBuffer = Buffer.from(maskResult.mask_b64, 'base64');
             const maskFileName = `mask_step_${combinedStep.step}.png`;

             // REMOVED: Local mask saving logic
             // const localMaskPath = path.join(localMaskDir, maskFileName);
             // try { ... } catch { ... }
             // ------------------------

             console.log(`Step ${combinedStep.step}: Uploading mask to storage...`);
             const maskFile = new File([maskBuffer], maskFileName, { type: 'image/png' });
             maskUrlForStep = await fal.storage.upload(maskFile);
             console.log(`Step ${combinedStep.step}: Mask uploaded successfully: ${maskUrlForStep}`);

             combinedExecutionPlan[i].generatedMaskUrl = maskUrlForStep;

         } catch (maskError) {
             console.error(`Step ${combinedStep.step}: Failed to generate or upload mask:`, maskError);
             return NextResponse.json(
                 { success: false, error: `Failed during mask generation for step ${combinedStep.step}: ${maskError instanceof Error ? maskError.message : String(maskError)}` },
                 { status: 500 }
             );
         }

         // --- Call Flux Inpainting ---
         if (maskUrlForStep) {
              console.log(`Step ${combinedStep.step}: Mask URL ${maskUrlForStep} ready. Calling Flux Inpainting...`);

              try {
                 // Define static parameters for Flux
                 const fluxParams = {
                     num_inference_steps: 28,
                     guidance_scale: 3.5,
                     real_cfg_scale: 3.5,
                     strength: 0.85, // Inpainting strength
                     num_images: 1,
                     enable_safety_checker: true,
                     reference_strength: 0.65,
                     reference_end: 1,
                     base_shift: 0.5,
                     max_shift: 1.15,
                     // controlnets and ip_adapters are typically empty unless specifically needed for inpaint style
                     controlnets: [],
                     ip_adapters: [],
                 };

                 // Prepare controlnet_unions payload according to the corrected schema
                 let fluxControlnetUnions: any[] = [];
                 const cnUnionDef = combinedStep.controlnet_unions?.[0]; // Get the union definition object from Gemini plan

                 // We need the path, depth URL, canny URL, and the scales from the definition
                 if (cnUnionDef && cnUnionDef.path && depthMapUrl && cannyMapUrl &&
                     Array.isArray(cnUnionDef.conditioning_scale) && cnUnionDef.conditioning_scale.length >= 2)
                 {
                      // Assume the first scale corresponds to depth, second to canny
                      const depthScale = cnUnionDef.conditioning_scale[0];
                      const cannyScale = cnUnionDef.conditioning_scale[1];

                      fluxControlnetUnions = [
                          {
                              path: cnUnionDef.path,
                              controls: [
                                  { // Depth Control Object
                                      control_image_url: depthMapUrl, // Direct URL string
                                      control_mode: "depth",          // String mode
                                      conditioning_scale: depthScale  // Corresponding scale
                                  },
                                  { // Canny Control Object
                                      control_image_url: cannyMapUrl, // Direct URL string
                                      control_mode: "canny",          // String mode
                                      conditioning_scale: cannyScale  // Corresponding scale
                                  }
                              ]
                          }
                      ];
                      console.log(`Step ${combinedStep.step}: Configured ControlNet Union with Depth & Canny.`);
                 } else {
                      console.warn(`Step ${combinedStep.step}: ControlNet Union definition missing or incomplete in execution plan. Proceeding without ControlNets.`);
                      console.warn("Definition received:", JSON.stringify(cnUnionDef));
                      console.warn("Needed URLs:", { depthMapUrl, cannyMapUrl });
                      fluxControlnetUnions = []; // Ensure it's an empty array if not configured
                 }

                 // Construct the full input payload
                 const fluxInput = {
                     image_url: currentImageUrl, // Use the output from the previous step
                     mask_url: maskUrlForStep,
                     prompt: combinedStep.prompt, // Use the step-specific prompt
                     // controlnet_unions: fluxControlnetUnions, // Temporarily commented out
                     ...fluxParams // Spread the static parameters (ensure 'controlnets' is removed from fluxParams)
                 };

                 console.log(`Step ${combinedStep.step}: Flux Input Payload:`, JSON.stringify(fluxInput, null, 2)); // Log payload for debugging

                 // Call Fal.ai Flux Inpainting
                 const fluxResult = await fal.subscribe("fal-ai/flux-general/inpainting", {
                     input: fluxInput,
                     logs: true,
                     onQueueUpdate: (update) => {
                         if (update.status === "IN_PROGRESS") {
                             update.logs.map((log) => log.message).forEach(msg => console.log(`[Flux Step ${combinedStep.step}]: ${msg}`));
                         }
                     },
                 });

                 // Process the result
                 const newImageUrl = (fluxResult?.data as any)?.images?.[0]?.url;
                 if (typeof newImageUrl !== 'string' || !newImageUrl) {
                     console.error("Flux response structure:", JSON.stringify(fluxResult, null, 2));
                     throw new Error("Inpainted image URL not found or invalid in Flux response");
                 }

                 console.log(`Step ${combinedStep.step}: Inpainting successful. New image URL: ${newImageUrl}`);

                 // Update currentImageUrl for the next iteration
                 currentImageUrl = newImageUrl;
                 finalGeneratedImageUrl = newImageUrl; // Update the final image tracker

              } catch (fluxError) {
                   console.error(`Step ${combinedStep.step}: Failed during Flux inpainting call:`, fluxError);
                   return NextResponse.json(
                       { success: false, error: `Failed during inpainting for step ${combinedStep.step}: ${fluxError instanceof Error ? fluxError.message : String(fluxError)}` },
                       { status: 500 }
                   );
              }

         } else {
              console.error(`Step ${combinedStep.step}: Mask URL is null, cannot proceed with inpainting.`);
               return NextResponse.json(
                  { success: false, error: `Failed to obtain mask URL for step ${combinedStep.step}. Cannot proceed.` },
                  { status: 500 }
              );
         }
         // --- End Flux Call ---

     } // End of execution loop

     console.log("\n--- Mask Generation & Execution Loop Finished ---");

     // --- FINAL REFINEMENT STEP (Image-to-Image) ---
     console.log("\n--- Starting Final Refinement Step (Image-to-Image) ---");
     if (currentImageUrl && currentImageUrl !== originalImageUrl) { // Check if any inpainting actually happened
         try {
             const finalPrompt = userPrompt; // Use the original user prompt
             const refinementInput = {
                 image_url: currentImageUrl, // Use the result from the last inpainting step
                 prompt: finalPrompt,
                 strength: 0.5,
                 max_shift: 1.15,
                 base_shift: 0.5,
                 num_images: 1,
                 controlnets: [],
                 ip_adapters: [],
                 reference_end: 1,
                 guidance_scale: 6.5,
                 real_cfg_scale: 3.5,
                 controlnet_unions: [],
                 reference_strength: 0.65,
                 num_inference_steps: 29,
                 enable_safety_checker: true
             };

             console.log("Calling final image-to-image refinement...");
             console.log("Refinement Payload:", JSON.stringify(refinementInput, null, 2));

             const refinementResult = await fal.subscribe("fal-ai/flux-general/image-to-image", {
                 input: refinementInput,
                 logs: true,
                 onQueueUpdate: (update) => {
                     if (update.status === "IN_PROGRESS") {
                         update.logs.map((log) => log.message).forEach(msg => console.log(`[Refinement Step]: ${msg}`));
                     }
                 },
             });

             const refinedImageUrl = (refinementResult?.data as any)?.images?.[0]?.url;
             if (typeof refinedImageUrl !== 'string' || !refinedImageUrl) {
                 console.error("Refinement response structure:", JSON.stringify(refinementResult, null, 2));
                 throw new Error("Refined image URL not found or invalid in refinement response");
             }

             console.log(`Final refinement successful. Final image URL: ${refinedImageUrl}`);
             finalGeneratedImageUrl = refinedImageUrl; // Update the final tracker

         } catch (refinementError) {
             console.error(`Failed during final refinement step:`, refinementError);
             // Decide if this is fatal. We still have the result from the last inpaint step.
             // For now, log the error but proceed with the pre-refinement image.
             console.warn(`Proceeding with pre-refinement image URL: ${finalGeneratedImageUrl}`);
             // Optionally, return an error or add a note to the response
         }
     } else {
         console.log("Skipping final refinement step as no inpainting was performed or initial image URL is missing.");
     }
     // --- END FINAL REFINEMENT STEP ---


    // --- Final Logging and Response ---
    console.log("\n--- Final Results (After Inpainting) ---");
    console.log({
      selectedAttemptNumber: bestAttempt?.attemptNumber,
      finalValidationStatus: bestAttempt?.validationStatus,
      finalValidationErrorCount: bestAttempt?.errorCount,
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
        // Log the plan *with generated mask URLs*
        combinedExecutionPlanWithMasks: combinedExecutionPlan,
        finalGeneratedImageUrl: finalGeneratedImageUrl, // Show the final image URL
      generatedFeedbacks: generatedFeedbacks
    });


    // Return data needed for potential client-side display or next steps
    return NextResponse.json({
      success: true,
      message: `Processing completed. Best attempt: #${bestAttempt?.attemptNumber ?? 'N/A'}. Inpainting finished. Final image URL: ${finalGeneratedImageUrl}`,
      data: {
        originalImageUrl,
        cannyMapUrl,
        depthMapUrl,
        imageDimensions,
        // Include the plan with generated mask URLs
        combinedExecutionPlan: combinedExecutionPlan,
        finalGeneratedImageUrl: finalGeneratedImageUrl, // The final output after all steps
        validationStatus: bestAttempt?.validationStatus ?? 'no_valid_attempt',
        validationErrors: bestAttempt?.validationErrors ?? [],
        attemptNumberSelected: bestAttempt?.attemptNumber
      }
    });

  } catch (error) {
    console.error("Error in generateImage main handler:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { success: false, error: `Failed to process request: ${errorMessage}` },
      { status: 500 }
    );
  }
} 