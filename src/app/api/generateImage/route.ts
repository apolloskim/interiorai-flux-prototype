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

    // Resize the depth map
    const resizedDepthBuffer = await resizeImage(result.data.image.url, originalWidth, originalHeight);
    
    // Upload the resized depth map
    const resizedDepthFile = new File([resizedDepthBuffer], 'resized-depth.jpg', { type: 'image/jpeg' });
    const resizedDepthUrl = await fal.storage.upload(resizedDepthFile);
    
    return resizedDepthUrl;

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
    const analysis = await analyzeImageWithVision(
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
    if (typeof analysis !== 'string' || !analysis) {
      console.error("Analysis result is not a valid string:", analysis);
      throw new Error("Failed to get valid analysis from vision model");
    }

    // Log all generated URLs and data
    console.log({
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
      furnishedImage: furnishedImageUrl,
      furnitureList: furnitureList,
      groundingData: groundingData
    });

    return NextResponse.json({ 
      success: true, 
      message: "Images processed and uploaded successfully",
    })
  } catch (error) {
    console.error("Error in generateImage:", error)
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    )
  }
} 