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

// Replace the existing analyzeImageWithVision function with a new one that supports both models
async function analyzeImageWithVision(
  imageUrl: string,
  depthUrl: string,
  cannyUrl: string,
  furnishedImageUrl: string,
  florenceResults: any,
  userPrompt: string,
  imageWidth: number,
  imageHeight: number,
  model: 'gpt4o' | 'gemini' = 'gemini' // Default to Gemini
) {
  if (model === 'gemini') {
    return analyzeImageWithGemini(
      imageUrl,
      depthUrl,
      cannyUrl,
      furnishedImageUrl,
      florenceResults,
      userPrompt,
      imageWidth,
      imageHeight
    );
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

export async function POST(request: NextRequest) {
  let structuredLayout: any[] | null = null;
  let executionPlan: any[] | null = null;
  let validationStatus: string = "not_run";
  let validationErrors: any[] = [];
  let finalAnalysisString: string | null = null; // Store the string that produced the final layout

  // Declare vars needed across attempts
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

  try {
    const formData = await request.formData();
    const image = formData.get("image") as File;
    userPrompt = (formData.get("prompt") || formData.get("text")) as string;
    const model = (formData.get("model") || 'gemini') as 'gpt4o' | 'gemini'; // Default to gemini

    if (!image || !userPrompt) {
      const error = !image ? "No image provided" : "No prompt provided";
      return NextResponse.json({ success: false, error }, { status: 400 });
    }
    console.log("Received prompt:", userPrompt);

    // --- Initial Setup ---
    const imageBuffer = Buffer.from(await image.arrayBuffer());
    ({ width: originalWidth, height: originalHeight } = await getImageDimensions(imageBuffer));
    console.log("Original image dimensions:", { width: originalWidth, height: originalHeight });
    originalImageUrl = await fal.storage.upload(image);

    // --- Generate Supporting Assets ---
    [cannyMapUrl, depthMapUrl] = await Promise.all([
      generateCannyMap(originalImageUrl, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl, originalWidth, originalHeight),
    ]);
    furnishedImageUrl = await generateFurnishedImage(originalImageUrl, userPrompt);
    furnitureList = await getFurnitureListFromSA2VA(furnishedImageUrl);
    groundingData = await getGroundingDataFromFlorence(furnishedImageUrl, furnitureList);
    florenceResults = groundingData.results; // Assuming results exist

    // --- Attempt 1: Call Gemini & Validate ---
    console.log("\n--- Attempt 1: Generating Initial Layout ---");
    let attempt1Prompt = prepareGeminiPrompt("", florenceResults, userPrompt, originalWidth, originalHeight);
    finalAnalysisString = await analyzeImageWithGemini(
        originalImageUrl, depthMapUrl, cannyMapUrl, furnishedImageUrl,
        florenceResults, attempt1Prompt, // Pass prepared prompt string directly
        originalWidth, originalHeight
    );
    ({ structuredLayout, executionPlan } = parseGeminiResponse(finalAnalysisString));

    if (structuredLayout) {
        ({ validationStatus, validationErrors } = await validateLayoutWithApi(
            structuredLayout, depthMapUrl, originalWidth, originalHeight
        ));
    } else {
        validationStatus = "parsing_failed"; // Indicate layout couldn't be parsed
        console.error("Skipping validation because initial layout parsing failed.");
    }

    // --- Attempt 2 (Retry if needed) ---
    if (validationStatus === 'failed' && validationErrors.length > 0) {
        console.log("\n--- Attempt 2: Retrying Layout Generation with Validation Feedback ---");

        // Format validation errors for the prompt
        const validationFeedback = `
ISSUE: The previously generated plan failed geometric validation.

VALIDATION ERRORS:
\`\`\`json
${JSON.stringify(validationErrors, null, 2)}
\`\`\`

REQUEST: Please generate a **revised** \`structured_layout\` and \`execution_plan\` JSON. Specifically **address the VALIDATION ERRORS listed above** by adjusting the \`bounding_box\` and/or \`spatial_anchor\` for the failed objects. **Pay close attention to the depth values indicated in the error messages relative to the expected plane ranges** to ensure realistic placement within the empty room's geometry (depth/canny maps) and resolve collisions. Adhere to all original requirements and guardrails. Ensure the output is ONLY the raw JSON.
`;
        // Prepare and call Gemini again
        let attempt2Prompt = prepareGeminiPrompt(validationFeedback, florenceResults, userPrompt, originalWidth, originalHeight);
        finalAnalysisString = await analyzeImageWithGemini( // Overwrite with retry result
            originalImageUrl, depthMapUrl, cannyMapUrl, furnishedImageUrl,
            florenceResults, attempt2Prompt, // Pass prepared prompt string directly
            originalWidth, originalHeight
        );
        // Parse the *new* response
        const { structuredLayout: layoutRetry, executionPlan: planRetry } = parseGeminiResponse(finalAnalysisString);

        // Validate the *new* layout
        if (layoutRetry) {
             // Update main variables with retry results BEFORE validation
             structuredLayout = layoutRetry;
             executionPlan = planRetry;
             // Validate the retry layout
            ({ validationStatus, validationErrors } = await validateLayoutWithApi(
                structuredLayout, depthMapUrl, originalWidth, originalHeight
            ));
             console.log(`Retry validation status: ${validationStatus}`);
        } else {
            validationStatus = "retry_parsing_failed"; // Indicate retry layout couldn't be parsed
            console.error("Skipping validation because retry layout parsing failed.");
            // Keep the results from the first attempt in this case? Or clear them?
            // Let's keep the first attempt's layout/plan but mark validation as failed.
            validationStatus = 'failed'; // Revert status as retry parse failed
            // Errors from first validation are already stored in validationErrors
        }
    }

    // --- Final Logging and Response ---
    console.log("\n--- Final Results ---");
    console.log({
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
      furnishedImage: furnishedImageUrl,
      furnitureList: furnitureList,
      groundingData: groundingData, // Maybe exclude large data from final log?
      structuredLayout: structuredLayout, // Final layout (either from attempt 1 or 2)
      executionPlan: executionPlan,     // Final plan (either from attempt 1 or 2)
      validationStatus: validationStatus, // Final validation status
      validationErrors: validationErrors  // Final validation errors (from last attempt)
      // finalAnalysisString: finalAnalysisString // Optional: log the raw string
    });

    return NextResponse.json({
      success: true, // API call itself succeeded, check validationStatus in data
      message: `Processing completed. Validation status: ${validationStatus}`,
      data: {
          originalImageUrl,
          cannyMapUrl,
          depthMapUrl,
          furnishedImageUrl,
          // furnitureList, // Maybe omit from response?
          // groundingData, // Maybe omit from response?
          structuredLayout,
          executionPlan,
          validationStatus,
          validationErrors
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