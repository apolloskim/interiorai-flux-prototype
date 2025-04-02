"use client"

import { useState, useEffect } from "react"
import { ImageUploader } from "@/components/image-uploader"
import { PromptInput } from "@/components/prompt-input"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [prompt, setPrompt] = useState("")
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isTransitioning, setIsTransitioning] = useState(false)

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

      // Call our API endpoint
      const result = await fetch('/api/generateImage', {
        method: 'POST',
        body: formData,
      })

      const data = await result.json()

      if (data.success) {
        // For now, we'll still use the placeholder image
        setGeneratedImage("/placeholder.svg?height=800&width=800")
      } else {
        console.error('Failed to generate image:', data.error)
      }
    } catch (error) {
      console.error('Error generating image:', error)
    } finally {
      // Add a small delay for the transition effect
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
                src={generatedImage || "/placeholder.svg"}
                alt="Generated interior design"
                className={cn(
                  "w-full h-auto object-cover transition-opacity duration-300",
                  isTransitioning ? "opacity-0" : "opacity-100",
                )}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                {isGenerating && (
                  <div className="bg-background/80 p-4 rounded-full">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                )}
              </div>
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

