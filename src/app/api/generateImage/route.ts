import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"
import sharp from 'sharp'
import OpenAI from 'openai'
import { analyzeImageWithGemini } from '@/app/utils/gemini'
import fs from 'fs'

if (!process.env.FAL_KEY) {
  throw new Error("FAL_KEY environment variable is not set")
}

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY environment variable is not set")
}

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

// NEW Interface for storing attempt results
interface AttemptResult {
  attemptNumber: number;
  structuredLayout: any[] | null; // Use specific type if available
  executionPlan: any[] | null; // Use specific type if available
  validationStatus: string; // Keep as string to handle various statuses
  validationErrors: ValidationError[] | null;
  errorCount: number;
  analysisString: string | null; // Store the raw analysis string for this attempt
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
    const resizedCannyBuffer = await resizeImage(result.data.image.url, originalWidth, originalHeight);
    
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
    const originalDepthUrl = result.data.image.url;
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
function parseGeminiResponse(analysisString: string | null | undefined): { structuredLayout: any[] | null, executionPlan: any[] | null } {
    let structuredLayout = null;
    let executionPlan = null;

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
    layout: any[] | null,
    depthMapUrl: string | null,
    imageWidth: number,
    imageHeight: number
): Promise<{ validationStatus: string, validationErrors: any[] }> {
    let validationErrors: any[] = [];
    let validationStatus: string = "not_run";

    if (layout && depthMapUrl && imageWidth && imageHeight) {
      const validationApiUrl = process.env.VALIDATION_API_URL || 'http://localhost:5001/validate_layout';
      console.log(`Calling validation API at: ${validationApiUrl}`);

      try {
        const validationResponse = await fetch(validationApiUrl, {
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
            validationErrors = validationResult.errors || [];
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
  // Variables to hold the results across attempts
  let structuredLayout: any[] | null = null;
  let executionPlan: any[] | null = null;
  let validationStatus: string = "not_run";
  let validationErrors: any[] = [];
  let finalAnalysisString: string | null = null; // Store the string that produced the final layout
  let generatedFeedbacks: string[] = []; // <<< NEW: Array to store feedback strings

  // Variables for data generated once
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
  let model: 'gpt4o' | 'gemini' = 'gemini'; // Default or from request

  // NEW: Store results of each attempt
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
    console.log("Original image dimensions:", { width: originalWidth, height: originalHeight });
    originalImageUrl = await fal.storage.upload(image);

    // --- Generate Supporting Assets (Done once) ---
    console.log("\n--- Generating Supporting Assets ---");
    [cannyMapUrl, depthMapUrl] = await Promise.all([
      generateCannyMap(originalImageUrl!, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl!, originalWidth, originalHeight),
    ]);
    furnishedImageUrl = await generateFurnishedImage(originalImageUrl!, userPrompt);
    furnitureList = await getFurnitureListFromSA2VA(furnishedImageUrl!);
    groundingData = await getGroundingDataFromFlorence(furnishedImageUrl!, furnitureList!);
    florenceResults = groundingData.results; // Assuming results exist
    console.log("--- Supporting Assets Generated ---");

    // --- Loop for Generation and Validation ---
    let currentValidationFeedback = ""; // Feedback for the *next* prompt
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`\n--- Attempt ${attempt} of ${maxAttempts}: Layout Generation ---`);

      // Vars for this specific attempt's results
      let currentAnalysisString: string | null = null;
      let currentLayout: any[] | null = null;
      let currentPlan: any[] | null = null;
      let currentValidationStatus: string = "not_run";
      let currentValidationErrors: ValidationError[] = [];

      // 1. Prepare Prompt
      let currentPrompt = prepareGeminiPrompt(
        currentValidationFeedback, // Use feedback from *previous* attempt
        florenceResults!,
        userPrompt,
        originalWidth,
        originalHeight
      );

      // 2. Call Gemini
      try {
        currentAnalysisString = await analyzeImageWithGemini(
          originalImageUrl!, depthMapUrl!, cannyMapUrl!, furnishedImageUrl!,
          florenceResults!, currentPrompt,
          originalWidth, originalHeight
        );
      } catch (geminiError: any) {
        console.error(`Attempt ${attempt}: Error calling Gemini - ${geminiError.message}`);
        currentValidationStatus = `attempt_${attempt}_gemini_error`;
        // Store this failed attempt result
        attemptResults.push({
            attemptNumber: attempt,
            structuredLayout: null,
            executionPlan: null,
            validationStatus: currentValidationStatus,
            validationErrors: [],
            errorCount: 999, // Assign high error count on Gemini failure
            analysisString: null,
        });
        if (attempt === maxAttempts) break;
        continue; // Try next attempt
      }

      // 3. Parse Response
      try {
           ({ structuredLayout: currentLayout, executionPlan: currentPlan } = parseGeminiResponse(currentAnalysisString));
           if (!currentLayout || !currentPlan) {
                throw new Error("Parsing failed or returned null layout/plan.");
           }
      } catch (parseError: any) {
          console.error(`Attempt ${attempt}: Parsing failed - ${parseError.message}`);
          currentValidationStatus = `attempt_${attempt}_parsing_failed`;
           // Store this failed attempt result
           attemptResults.push({
                attemptNumber: attempt,
                structuredLayout: null, // Parsed layout is null
                executionPlan: null, // Parsed plan is null
                validationStatus: currentValidationStatus,
                validationErrors: [],
                errorCount: 998, // Assign high error count on parse failure
                analysisString: currentAnalysisString, // Store the string that failed
           });
          if (attempt === maxAttempts) break;
          continue; // Try next attempt
      }

      // 4. Validate Layout
      console.log(`--- Attempt ${attempt}: Validating Layout ---`);
      ({ validationStatus: currentValidationStatus, validationErrors: currentValidationErrors } = await validateLayoutWithApi(
        currentLayout, depthMapUrl!, originalWidth, originalHeight
      ));

      // 5. Store Result of This Attempt
      const currentErrorCount = currentValidationErrors?.length ?? 0;
      attemptResults.push({
        attemptNumber: attempt,
        structuredLayout: currentLayout,
        executionPlan: currentPlan,
        validationStatus: currentValidationStatus,
        validationErrors: currentValidationErrors,
        errorCount: currentErrorCount,
        analysisString: currentAnalysisString,
      });

      // 6. Check Validation and Loop Control
      if (currentValidationStatus === 'success') {
        console.log(`Attempt ${attempt}: Validation successful!`);
        break; // Exit loop on success
      } else {
        console.warn(`Attempt ${attempt}: Validation status: ${currentValidationStatus}.`);
        if (attempt < maxAttempts) {
          // Prepare feedback for the *next* iteration
          console.log(`Generating intelligent feedback for attempt ${attempt + 1}...`);
          const generatedFeedback = generateRepromptFeedback(currentValidationErrors);

          if (generatedFeedback) {
            currentValidationFeedback = generatedFeedback; // Set feedback for the next loop
            generatedFeedbacks.push(generatedFeedback); // Log feedback history
          } else {
            console.warn(`Attempt ${attempt}: Validation failed, but no actionable feedback generated. Stopping retries.`);
            currentValidationFeedback = ""; // Clear feedback
            break; // Exit loop
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
      // Prioritize success first
      const successfulAttempts = attemptResults.filter(r => r.validationStatus === 'success');
      if (successfulAttempts.length > 0) {
        bestAttempt = successfulAttempts[0]; // Take the first success
        console.log(`Selected best attempt: #${bestAttempt.attemptNumber} (Validation Passed)`);
      } else {
        // No success, find the one with the minimum number of errors
        // Filter out attempts that failed due to Gemini/Parsing errors (high error counts) unless they are the *only* results
        const validAttempts = attemptResults.filter(r => r.errorCount < 900);
         if (validAttempts.length > 0) {
            validAttempts.sort((a, b) => a.errorCount - b.errorCount);
            bestAttempt = validAttempts[0];
             console.log(`Selected best attempt: #${bestAttempt.attemptNumber} (Failed Validation, Min Errors: ${bestAttempt.errorCount})`);
         } else {
             // If only Gemini/Parsing errors occurred, maybe pick the last attempt? Or none?
             // Let's pick the last one stored for now, even if it failed early.
             bestAttempt = attemptResults[attemptResults.length - 1];
             console.warn(`Selected best attempt: #${bestAttempt.attemptNumber} (Failed early: ${bestAttempt.validationStatus}). Review manually.`);
         }
      }
    } else {
      console.log("No attempts were completed or stored.");
    }


    // --- Final Logging and Response ---
    console.log("\n--- Final Results (Based on Best Attempt) ---");
    // Log details from the 'bestAttempt' object
    console.log({
      selectedAttemptNumber: bestAttempt?.attemptNumber,
      finalValidationStatus: bestAttempt?.validationStatus,
      finalValidationErrorCount: bestAttempt?.errorCount,
      finalValidationErrors: bestAttempt?.validationErrors,
      // Log assets generated initially
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
      furnishedImage: furnishedImageUrl,
      // Log the selected layout/plan
      structuredLayout: bestAttempt?.structuredLayout,
      executionPlan: bestAttempt?.executionPlan,
      // Log feedback history
      generatedFeedbacks: generatedFeedbacks
      // bestAttemptAnalysisString: bestAttempt?.analysisString // Optional: Log the raw string of the best attempt
    });

    // Return data from the 'bestAttempt'
    return NextResponse.json({
      success: true, // API call itself succeeded
      message: `Processing completed. Best attempt: #${bestAttempt?.attemptNumber ?? 'N/A'}, Final validation status: ${bestAttempt?.validationStatus ?? 'unknown'}`,
      data: {
        originalImageUrl,
        cannyMapUrl,
        depthMapUrl,
        furnishedImageUrl,
        structuredLayout: bestAttempt?.structuredLayout ?? null, // Use null if no best attempt
        executionPlan: bestAttempt?.executionPlan ?? null,
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