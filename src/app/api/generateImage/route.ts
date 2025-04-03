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
    const result = await fal.subscribe("fal-ai/imageutils/depth", {
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


// Replace the existing analyzeImageWithVision function with a new one that supports both models
async function analyzeImageWithVision(
  imageUrl: string,
  cannyUrl: string,
  depthUrl: string,
  userPrompt: string,
  imageWidth: number,
  imageHeight: number,
  model: 'gpt4o' | 'gemini' = 'gemini' // Default to Gemini
) {
  if (model === 'gemini') {
    return analyzeImageWithGemini(
      imageUrl,
      cannyUrl,
      depthUrl,
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

    // Analyze the image using GPT-4o with all context
    const analysis = await analyzeImageWithVision(
      originalImageUrl,
      cannyMapUrl,
      depthMapUrl,
      prompt,
      originalWidth,
      originalHeight,
      model
    );

    // Log all generated URLs
    console.log({
      originalImage: originalImageUrl,
      cannyMap: cannyMapUrl,
      depthMap: depthMapUrl,
      analysis
    });

    return NextResponse.json({ 
      success: true, 
      message: "Images processed and uploaded successfully"
    })
  } catch (error) {
    console.error("Error in generateImage:", error)
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    )
  }
} 