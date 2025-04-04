import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"
import sharp from 'sharp'
import OpenAI from 'openai'
import { analyzeImageWithGemini } from '@/app/utils/gemini'

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

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get("image") as File
    const prompt = (formData.get("prompt") || formData.get("text")) as string
    const model = (formData.get("model") || 'gpt4o') as 'gpt4o' | 'gemini'

    if (!image) {
      return NextResponse.json(
        { success: false, error: "No image provided" },
        { status: 400 }
      )
    }

    if (!prompt) {
      return NextResponse.json(
        { success: false, error: "No prompt provided" },
        { status: 400 }
      )
    }

    console.log("Received prompt:", prompt);

    // Get original image dimensions and create gridded version
    const imageBuffer = Buffer.from(await image.arrayBuffer());
    const { width: originalWidth, height: originalHeight } = await getImageDimensions(imageBuffer);
    console.log("Original image dimensions:", { width: originalWidth, height: originalHeight });

    // Upload original image
    const originalImageUrl = await fal.storage.upload(image);
    
    // First generate maps
    const [cannyMapUrl, depthMapUrl] = await Promise.all([
      generateCannyMap(originalImageUrl, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl, originalWidth, originalHeight),
    ]);

    // Generate the final furnished image using the analysis as the prompt
    const furnishedImageUrl = await generateFurnishedImage(originalImageUrl, prompt);

    // Get the furniture list from SA2VA
    const furnitureList = await getFurnitureListFromSA2VA(furnishedImageUrl);

    // Get grounding data from Florence-2
    const groundingData = await getGroundingDataFromFlorence(furnishedImageUrl, furnitureList);
    const florenceResults = groundingData.results;
    

    // // Analyze the image using GPT-4o with all context
    const analysisString = await analyzeImageWithVision(
      originalImageUrl,
      depthMapUrl,
      cannyMapUrl,
      furnishedImageUrl,
      florenceResults,
      prompt,
      originalWidth,
      originalHeight,
      model
    );

    // // Ensure analysis is valid before proceeding
    if (typeof analysisString !== 'string' || !analysisString) {
      console.error("Analysis result is not a valid string:", analysisString);
      throw new Error("Failed to get valid analysis string from vision model");
    }
    
    // --- NEW: Parse the analysis string and extract structured_layout ---
    let structuredLayout = null;
    let executionPlan = null;
    let analysisJson = null; // To hold the full parsed object

    try {
      // --- UPDATED CLEANING LOGIC ---
      let jsonStringToParse = analysisString; // Assume no fences initially
      const match = analysisString.match(/^```json\s*([\s\S]*?)\s*```$/); 
      // Check if the regex matched and captured content (group 1)
      if (match && match[1]) {
        jsonStringToParse = match[1]; // Use only the captured content
        console.log("Extracted JSON content from fences.");
      } else {
        console.log("No JSON fences found, attempting to parse original string.");
      }
      // --- END UPDATED CLEANING LOGIC ---
      
      // Parse the potentially cleaned string
      analysisJson = JSON.parse(jsonStringToParse); 
      
      if (analysisJson && typeof analysisJson === 'object') {
        structuredLayout = analysisJson.structured_layout;
        executionPlan = analysisJson.execution_plan; // Also extract execution_plan if needed later
        
        if (!Array.isArray(structuredLayout)) {
            console.error("Parsed analysis does not contain a valid 'structured_layout' array:", analysisJson);
            structuredLayout = null; // Reset if invalid
        } else {
            console.log("Successfully extracted structured_layout:", structuredLayout);
        }
        // Optional: Add similar validation for executionPlan if you use it
        
      } else {
         console.error("Parsed analysis is not a valid object:", analysisJson);
      }
      
    } catch (parseError) {
      console.error("Failed to parse analysis string as JSON:", parseError);
      console.error("Original analysis string:", analysisString);
      // Decide how to handle the error, maybe return an error response or proceed without the layout
      throw new Error("Failed to parse layout data from vision model response."); 
    }
    // --- END NEW ---

    // --- NEW: Call Flask Validation API ---
    let validationErrors: any[] = []; // Initialize as empty array
    let validationStatus: string = "not_run"; // Track status

    if (structuredLayout && depthMapUrl && originalWidth && originalHeight) {
      const validationApiUrl = process.env.VALIDATION_API_URL || 'http://localhost:5001/validate_layout'; // Use env var or default
      console.log(`Calling validation API at: ${validationApiUrl}`);

      try {
        const validationResponse = await fetch(validationApiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            layout: structuredLayout,
            image_dimensions: { width: originalWidth, height: originalHeight },
            depth_map_url: depthMapUrl, // Send the generated depth map URL
          }),
          // Add a timeout (optional but recommended)
          signal: AbortSignal.timeout(30000) // 30 seconds timeout
        });

        if (!validationResponse.ok) {
          // Handle HTTP errors from Flask (e.g., 400, 500)
          const errorBody = await validationResponse.text(); // Read error body as text first
          console.error(`Validation API returned error ${validationResponse.status}: ${errorBody}`);
          validationStatus = "api_error";
          // Optionally, rethrow or add a generic error to validationErrors
          // validationErrors.push({ check: "API", message: `Validation API failed with status ${validationResponse.status}`}); 
        } else {
          const validationResult = await validationResponse.json();
          if (validationResult.status === 'success') {
            validationStatus = "success";
            console.log("Layout validation successful.");
          } else if (validationResult.status === 'error') {
            validationStatus = "failed";
            validationErrors = validationResult.errors || []; // Ensure errors is an array
            console.warn(`Layout validation failed with ${validationErrors.length} errors.`);
            console.warn("Validation Errors:", JSON.stringify(validationErrors, null, 2)); 
          } else {
             validationStatus = "unknown_response";
             console.error("Validation API returned unexpected status:", validationResult);
          }
        }
      } catch (validationError) {
         validationStatus = "network_error";
         if (validationError instanceof Error && validationError.name === 'TimeoutError') {
             console.error("Validation API call timed out:", validationError);
         } else {
            console.error("Error calling validation API:", validationError);
         }
         // Optionally, add a generic error to validationErrors
         // validationErrors.push({ check: "Network", message: "Failed to connect to validation API." });
      }
    } else {
        console.warn("Skipping validation API call due to missing data (layout, depth map, or dimensions).");
        validationStatus = "skipped";
    }
    // --- END NEW ---

    // Log all generated URLs and data
    console.log({
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
      furnishedImage: furnishedImageUrl,
      furnitureList: furnitureList,
      groundingData: groundingData,
      structuredLayout: structuredLayout, 
      executionPlan: executionPlan,     
      validationStatus: validationStatus, // Log validation status
      validationErrors: validationErrors  // Log validation errors
    });

    // --- UPDATE RESPONSE: Include validation results ---
    return NextResponse.json({ 
      success: true, 
      message: "Images processed, analyzed, and validated successfully", // Updated message
      data: {
          originalImageUrl,
          cannyMapUrl,
          depthMapUrl,
          furnishedImageUrl,
          furnitureList,
          groundingData,
          structuredLayout, 
          executionPlan,
          validationStatus, // Include validation status
          validationErrors  // Include validation errors
      }
    });
  } catch (error) {
    console.error("Error in generateImage:", error);
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      { success: false, error: `Failed to process request: ${errorMessage}` },
      { status: 500 }
    );
  }
} 