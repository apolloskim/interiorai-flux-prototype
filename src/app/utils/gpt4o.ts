import OpenAI from 'openai';
import fs from 'fs';
import path from 'path';

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY environment variable is not set");
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface GPT4oInput {
  originalImage: string;  // URL
  griddedOriginalImage: string;  // URL
  cannyMap: string;  // URL
  griddedCannyMap: string;  // URL
  depthMap: string;  // URL
  griddedDepthMap: string;  // URL
  furnishedImage: string;  // URL
  florenceJson: any;  // Object from Florence detection
  userPrompt: string;
  imageResolution: string;  // e.g. "1024x1024"
}

type ImageDetail = "high" | "low" | "auto";

export async function generateLayoutPlan(input: GPT4oInput) {
  try {
    // Read the prompt template
    const promptTemplate = fs.readFileSync(
      path.join(process.cwd(), 'src/app/utils/goated-gpt-4o-prompt.txt'),
      'utf8'
    );

    // Replace placeholders in the prompt
    const prompt = promptTemplate
      .replace('[PASTE FLORENCE JSON HERE]', JSON.stringify(input.florenceJson, null, 2))
      .replace('[PASTE USER PROMPT HERE]', input.userPrompt)
      .replace('[Placeholder: Image resolution, e.g. 1024x768]', input.imageResolution);

    // Prepare the messages array with images
    const messages = [
      {
        role: "user" as const,
        content: [
          {
            type: "text" as const,
            text: prompt
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.originalImage,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.griddedOriginalImage,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.furnishedImage,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.cannyMap,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.griddedCannyMap,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.depthMap,
              detail: "high" as ImageDetail
            }
          },
          {
            type: "image_url" as const,
            image_url: {
              url: input.griddedDepthMap,
              detail: "high" as ImageDetail
            }
          }
        ]
      }
    ];

    console.log(`#### messages: ${JSON.stringify(messages)}`);

    // Call GPT-4o
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-2024-11-20", // don't change this
      messages,
      max_tokens: 4096,
      temperature: 0.2,
    });

    console.log(`####completion: ${JSON.stringify(completion)}`);

    // Parse and return the JSON response
    const responseText = completion.choices[0]?.message?.content;
    if (!responseText) {
      throw new Error("No response from GPT-4o");
    }

    try {
      return JSON.parse(responseText);
    } catch (error) {
      console.error("Failed to parse GPT-4o response as JSON:", responseText);
      throw new Error("Invalid JSON response from GPT-4o");
    }
  } catch (error) {
    console.error("Error in generateLayoutPlan:", error);
    throw error;
  }
} 