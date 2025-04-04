import { GoogleGenAI } from "@google/genai";
import fs from 'fs';

if (!process.env.GEMINI_API_KEY) {
  throw new Error("GEMINI_API_KEY environment variable is not set");
}

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY, httpOptions: { apiVersion: 'v1alpha' } });

// Helper function to fetch image and convert to base64
async function imageUrlToBase64(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch image from ${url}: ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer).toString('base64');
}

export async function analyzeImageWithGemini(
  imageUrl: string, 
  depthUrl: string, 
  cannyUrl: string, 
  furnishedImageUrl: string,
  florenceResults: any,
  userPrompt: string, 
  imageWidth: number, 
  imageHeight: number
) {
  try {
    // Convert all images to base64 first to handle any potential failures early
    const [
      image1, image2, image3, image4
    ] = await Promise.all([
      imageUrlToBase64(imageUrl),
      imageUrlToBase64(depthUrl),
      imageUrlToBase64(cannyUrl),
      imageUrlToBase64(furnishedImageUrl),
    ]);

    // Read and prepare the prompt template
    const promptTemplate = fs.readFileSync('src/app/utils/goated-gemini-prompt.txt', 'utf8');
    const filledPrompt = promptTemplate
        .replace('[PLACEHOLDER FOR IMAGE_DIMENSIONS]', `${imageWidth}x${imageHeight}`)
        .replace('[PLACEHOLDER FOR USER_STYLE_PROMPT]', userPrompt)
        .replace('[PLACEHOLDER FOR FLORENCE_JSON]', JSON.stringify(florenceResults));

    // Following documentation best practice: put images first for better results
    const contents = [
        // Images first
        { inlineData: { data: image1, mimeType: "image/jpeg" } },          // Image A: Original empty interior        // Image C: Hallucinated furnished interior
        { inlineData: { data: image2, mimeType: "image/jpeg" } },          // Image B: Depth (non-gridded)
        { inlineData: { data: image3, mimeType: "image/jpeg" } },          // Image C: Canny (non-gridded)
        { inlineData: { data: image4, mimeType: "image/jpeg" } },          // Image D: Furnished image
        // Text prompt last
        { text: filledPrompt }
    ];

    // Generate content and handle response
    const result = await genAI.models.generateContent({
      model: "gemini-2.5-pro-exp-03-25",
      contents: contents
    });

    if (!result.candidates?.[0]?.content?.parts?.[0]?.text) {
      throw new Error("No valid text response received from Gemini");
    }
    
    const text = result.candidates[0].content.parts[0].text;

    if (process.env.NODE_ENV === 'development') {
      console.log("Filled Prompt:", filledPrompt);
      console.log("Gemini Response:", text);
    }

    return text;
  } catch (error: any) {
    console.error("Error in Gemini analysis:", error);
    // Re-throw with more context
    throw new Error(`Failed to analyze images with Gemini: ${error.message}`);
  }
} 