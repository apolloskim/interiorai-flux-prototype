"use client"

import { useState, useEffect } from "react"
import { ImageUploader } from "@/components/image-uploader"
import { PromptInput } from "@/components/prompt-input"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [prompt, setPrompt] = useState("")
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isTransitioning, setIsTransitioning] = useState(false)
  const [selectedModel, setSelectedModel] = useState<'gpt4o' | 'gemini'>('gpt4o')

  const handleGenerate = async () => {
    if (!uploadedImage || !prompt.trim()) return

    setIsGenerating(true)
    setIsTransitioning(true)

    try {
      // Convert base64 to blob
      const response = await fetch(uploadedImage)
      const blob = await response.blob()
      
      // Create form data
      const formData = new FormData()
      formData.append('image', blob, 'image.jpg')
      formData.append('text', prompt)
      formData.append('model', selectedModel)

      // Call our API endpoint
      const result = await fetch('/api/generateImage', {
        method: 'POST',
        body: formData,
      })

      const data = await result.json()

      if (data.success && data.data?.finalGeneratedImageUrl) {
        console.log("Generated Image URL:", data.data.finalGeneratedImageUrl);
        // Use the actual final image URL from the API response
        setGeneratedImage(data.data.finalGeneratedImageUrl);
      } else {
        console.error('Failed to generate image or URL missing:', data.error || 'No final image URL found');
        // Optionally set an error state or keep generatedImage null
        setGeneratedImage(null); // Ensure it's null on failure
      }
    } catch (error) {
      console.error('Error generating image:', error)
      setGeneratedImage(null); // Ensure it's null on catch
    } finally {
      // Add a small delay for the transition effect, even on error
      setTimeout(() => {
        setIsGenerating(false)
        setIsTransitioning(false)
      }, 300)
    }
  }

  // Reset generated image when uploaded image is removed
  useEffect(() => {
    if (!uploadedImage) {
      setGeneratedImage(null)
    }
  }, [uploadedImage])

  return (
    <main className="flex flex-col md:flex-row h-screen bg-background">
      {/* Left Panel - Input */}
      <div className="w-full md:w-1/3 p-6 border-r overflow-y-auto">
        <div className="space-y-6">
          <div className="space-y-2">
            <h1 className="text-2xl font-bold">InteriorAI Fal Prototype</h1>
            <p className="text-sm text-muted-foreground">
              Transform your interior spaces with AI-powered design suggestions
            </p>
          </div>

          <div className="space-y-6">
            <ImageUploader onImageUploaded={setUploadedImage} uploadedImage={uploadedImage} compact={true} />

            <div className="space-y-2">
              <Label className="text-sm font-medium">Model Selection</Label>
              <div className="flex space-x-4">
                <label className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="model"
                    value="gpt4o"
                    checked={selectedModel === 'gpt4o'}
                    onChange={(e) => setSelectedModel(e.target.value as 'gpt4o' | 'gemini')}
                    className="h-4 w-4 text-primary"
                  />
                  <span className="text-sm">GPT-4o</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="model"
                    value="gemini"
                    checked={selectedModel === 'gemini'}
                    onChange={(e) => setSelectedModel(e.target.value as 'gpt4o' | 'gemini')}
                    className="h-4 w-4 text-primary"
                  />
                  <span className="text-sm">Gemini</span>
                </label>
              </div>
            </div>

            <PromptInput
              value={prompt}
              onChange={setPrompt}
              disabled={!uploadedImage || isGenerating}
              placeholder="Describe the interior style you want (e.g., 'modern minimalist living room with plants')..."
              hint={
                <p className="text-xs text-muted-foreground">
                  Be specific about interior styles, furniture, colors, and materials you'd like to see.
                </p>
              }
            />

            <Button
              onClick={handleGenerate}
              disabled={!uploadedImage || !prompt.trim() || isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Image"
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Right Panel - Output */}
      <div className="w-full md:w-2/3 p-6 bg-muted/10 flex items-center justify-center">
        <div className="w-full max-w-3xl">
          {generatedImage ? (
            <div className="relative rounded-lg overflow-hidden shadow-lg">
              <img
                src={generatedImage}
                alt="Generated interior design"
                className={cn(
                  "w-full h-auto object-cover transition-opacity duration-300",
                  isTransitioning ? "opacity-0" : "opacity-100",
                )}
                onError={(e) => {
                   console.error("Failed to load generated image:", generatedImage);
                   e.currentTarget.src = "/placeholder.svg";
                   e.currentTarget.alt = "Failed to load generated image";
                 }}
              />
              {isGenerating && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/50">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              )}
            </div>
          ) : (
            <div className="border-2 border-dashed border-muted-foreground/20 rounded-lg p-12 text-center h-[500px] flex flex-col items-center justify-center">
              <p className="text-muted-foreground">
                {uploadedImage
                  ? "Click 'Generate Image' to see the result here"
                  : "Upload an image and enter a prompt to get started"}
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

