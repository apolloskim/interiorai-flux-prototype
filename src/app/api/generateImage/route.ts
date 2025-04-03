import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"
import sharp from 'sharp'
import OpenAI from 'openai'
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

async function createGridOverlay(width: number, height: number): Promise<Buffer> {
  // Create SVG with 50px grid lines but 100px coordinate labels
  const svg = `
    <svg width="${width}" height="${height}">
      <defs>
        <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
          <path d="M 50 0 L 0 0 0 50" fill="none" stroke="rgba(200,200,200,0.5)" stroke-width="0.5"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />
      
      <!-- Add numbers every 100px on the top -->
      ${Array.from({ length: Math.floor(width / 100) }, (_, i) => 
        `<text x="${i * 100 + 2}" y="10" fill="rgba(200,200,200,0.8)" font-size="10">${i * 100}</text>`
      ).join('')}
      
      <!-- Add numbers every 100px on the left -->
      ${Array.from({ length: Math.floor(height / 100) }, (_, i) => 
        `<text x="2" y="${i * 100 + 10}" fill="rgba(200,200,200,0.8)" font-size="10">${i * 100}</text>`
      ).join('')}
    </svg>`;

  return Buffer.from(svg);
}

async function addGridToImage(imageBuffer: Buffer): Promise<Buffer> {
  const { width, height } = await getImageDimensions(imageBuffer);
  const gridOverlay = await createGridOverlay(width, height);
  
  return await sharp(imageBuffer)
    .composite([
      {
        input: gridOverlay,
        top: 0,
        left: 0,
      }
    ])
    .toBuffer();
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

async function generateCannyMap(imageUrl: string, originalWidth: number, originalHeight: number): Promise<{ original: string; gridded: string }> {
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

    // Add grid to the resized canny map and upload
    const griddedCannyBuffer = await addGridToImage(resizedCannyBuffer);
    const griddedCannyFile = new File([griddedCannyBuffer], 'gridded-canny.jpg', { type: 'image/jpeg' });
    const griddedCannyUrl = await fal.storage.upload(griddedCannyFile);
    
    return {
      original: resizedCannyUrl,
      gridded: griddedCannyUrl
    };
  } catch (error) {
    console.error("Error generating canny map:", error);
    throw error;
  }
}

async function generateDepthMap(imageUrl: string, originalWidth: number, originalHeight: number): Promise<{ original: string; gridded: string }> {
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

    // Add grid to the resized depth map and upload
    const griddedDepthBuffer = await addGridToImage(resizedDepthBuffer);
    const griddedDepthFile = new File([griddedDepthBuffer], 'gridded-depth.jpg', { type: 'image/jpeg' });
    const griddedDepthUrl = await fal.storage.upload(griddedDepthFile);
    
    return {
      original: resizedDepthUrl,
      gridded: griddedDepthUrl
    };
  } catch (error) {
    console.error("Error generating depth map:", error);
    throw error;
  }
}

async function generateFurnishedImage(imageUrl: string, prompt: string): Promise<any> {
  try {
    const result = await fal.subscribe("fal-ai/flux-general/image-to-image", {
      input: {
        loras: [],
        prompt,
        strength: 0.85,
        image_url: imageUrl,
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

    console.log("Flux generation result:", result.data);
    return result.data;
  } catch (error) {
    console.error("Error generating furnished image:", error);
    throw error;
  }
}

async function detectObjects(imageUrl: string): Promise<any> {
  try {
    const result = await fal.subscribe(
      "fal-ai/florence-2-large/object-detection",
      {
        input: {
          image_url: imageUrl
        },
        logs: true,
        onQueueUpdate: (update) => {
          if (update.status === "IN_PROGRESS") {
            update.logs.map((log) => log.message).forEach(console.log);
          }
        },
      }
    );

    console.log("Object detection result:", result.data);
    return result.data;
  } catch (error) {
    console.error("Error detecting objects:", error);
    throw error;
  }
}

async function analyzeImageWithVision(imageUrl: string, griddedImageUrl: string, cannyUrl: string, griddedCannyUrl: string, depthUrl: string, griddedDepthUrl: string, furnishedImageUrl: string, florenceJson: any, userPrompt: string, imageWidth: number, imageHeight: number) {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('Missing OpenAI API key');
  }

  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  // Read the prompt template
  const promptTemplate = fs.readFileSync('src/app/utils/goated-gpt-4o-prompt.txt', 'utf8');
  
  // Replace placeholders with actual values
  const filledPrompt = promptTemplate
    .replace('[Placeholder: Image resolution, e.g. 1024x768]', `${imageWidth}x${imageHeight}`)
    .replace('[PASTE FLORENCE JSON HERE]', JSON.stringify(florenceJson, null, 2))
    .replace('[PASTE USER PROMPT HERE]', userPrompt)
    .replace('[placeholder above]', `${imageWidth}x${imageHeight}`)

  console.log(`#### filledPrompt: ${filledPrompt}\n\n\n\n`);

  const response = await openai.responses.create({
    model: "gpt-4o-2024-11-20", // don't change this
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: filledPrompt },
          {
            type: "input_image",
            image_url: imageUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: griddedImageUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: cannyUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: griddedCannyUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: depthUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: griddedDepthUrl,
            detail: "high"
          },
          {
            type: "input_image",
            image_url: furnishedImageUrl,
            detail: "high"
          }
        ]
      }
    ]
  });

  console.log(`#### response: ${JSON.stringify(response)}\n\n\n\n`);

  console.log("GPT-4o Response:", response.output_text);
  return response.output_text;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get("image") as File
    const prompt = (formData.get("prompt") || formData.get("text")) as string

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

    // Create and upload gridded version
    const griddedImageBuffer = await addGridToImage(imageBuffer);
    const griddedImageFile = new File([griddedImageBuffer], 'gridded-' + image.name, { type: 'image/jpeg' });
    const griddedImageUrl = await fal.storage.upload(griddedImageFile);
    
    // First generate maps and furnished image
    const [cannyMaps, depthMaps, furnishedImageData] = await Promise.all([
      generateCannyMap(originalImageUrl, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl, originalWidth, originalHeight),
      generateFurnishedImage(originalImageUrl, prompt)
    ]);

    // Then use the furnished image URL for object detection
    const furnishedImageUrl = furnishedImageData.images[0].url;
    const objectDetectionData = await detectObjects(furnishedImageUrl);

    // Analyze the image using GPT-4o with all context
    const analysis = await analyzeImageWithVision(
      originalImageUrl,
      griddedImageUrl,
      cannyMaps.original,
      cannyMaps.gridded,
      depthMaps.original,
      depthMaps.gridded,
      furnishedImageUrl,
      objectDetectionData,
      prompt,
      originalWidth,
      originalHeight
    );

    // Log all generated URLs
    console.log({
      originalImage: originalImageUrl,
      griddedImage: griddedImageUrl,
      cannyMap: cannyMaps.original,
      griddedCannyMap: cannyMaps.gridded, 
      depthMap: depthMaps.original,
      griddedDepthMap: depthMaps.gridded,
      furnishedImage: furnishedImageUrl,
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