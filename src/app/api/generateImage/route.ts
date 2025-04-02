import { NextRequest, NextResponse } from "next/server"
import { fal } from "@fal-ai/client"

if (!process.env.FAL_KEY) {
  throw new Error("FAL_KEY environment variable is not set")
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const image = formData.get("image") as File
    const text = formData.get("text")

    // Log the received data
    console.log("Received text:", text)

    if (!image) {
      return NextResponse.json(
        { success: false, error: "No image provided" },
        { status: 400 }
      )
    }

    // Upload the image to fal
    const imageUrl = await fal.storage.upload(image)
    console.log("Uploaded image URL:", imageUrl)

    return NextResponse.json({ 
      success: true, 
      message: "Image and text received successfully",
      imageUrl: imageUrl
    })
  } catch (error) {
    console.error("Error in generateImage:", error)
    return NextResponse.json(
      { success: false, error: "Failed to process request" },
      { status: 500 }
    )
  }
} 