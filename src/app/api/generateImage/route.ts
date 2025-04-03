import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"
import sharp from 'sharp'

if (!process.env.FAL_KEY) {
  throw new Error("FAL_KEY environment variable is not set")
}

async function getImageDimensions(buffer: Buffer): Promise<{ width: number; height: number }> {
  const metadata = await sharp(buffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error("Could not get image dimensions");
  }
  return { width: metadata.width, height: metadata.height };
}

async function createGridOverlay(width: number, height: number, gridSize: number = 50): Promise<Buffer> {
  // Create SVG with grid lines
  const svg = `
    <svg width="${width}" height="${height}">
      <defs>
        <pattern id="grid" width="${gridSize}" height="${gridSize}" patternUnits="userSpaceOnUse">
          <path d="M ${gridSize} 0 L 0 0 0 ${gridSize}" fill="none" stroke="rgba(255,255,255,0.5)" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />
      
      <!-- Add numbers on the top -->
      ${Array.from({ length: Math.floor(width / gridSize) }, (_, i) => 
        `<text x="${i * gridSize + 5}" y="20" fill="white" font-size="12">${i * gridSize}</text>`
      ).join('')}
      
      <!-- Add numbers on the left -->
      ${Array.from({ length: Math.floor(height / gridSize) }, (_, i) => 
        `<text x="5" y="${i * gridSize + 20}" fill="white" font-size="12">${i * gridSize}</text>`
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

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get("image") as File
    // Check for either prompt or text in the form data
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
    
    // Generate both maps and furnished image in parallel using original image
    const [cannyMapUrl, depthMapUrl, furnishedImageData] = await Promise.all([
      generateCannyMap(originalImageUrl, originalWidth, originalHeight),
      generateDepthMap(originalImageUrl, originalWidth, originalHeight),
      generateFurnishedImage(originalImageUrl, prompt)
    ])

    console.log("Original image URL:", originalImageUrl)
    console.log("Gridded image URL:", griddedImageUrl)
    console.log("Resized Canny map URL:", cannyMapUrl)
    console.log("Resized Depth map URL:", depthMapUrl)
    console.log("Furnished image data:", JSON.stringify(furnishedImageData, null, 2))

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