"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ImageUploaderProps {
  onImageUploaded: (imageUrl: string | null) => void
  uploadedImage: string | null
  compact?: boolean
}

export function ImageUploader({ onImageUploaded, uploadedImage, compact = false }: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (file: File) => {
    if (!file.type.match("image.*")) return

    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        onImageUploaded(e.target.result as string)
      }
    }
    reader.readAsDataURL(file)
  }

  const handleRemoveImage = () => {
    onImageUploaded(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className="space-y-3">
      <h2 className="text-sm font-medium">Upload Interior Image</h2>

      {!uploadedImage ? (
        <div
          className={cn(
            "border-2 border-dashed rounded-lg text-center transition-colors cursor-pointer",
            isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/20",
            compact ? "p-4" : "p-8",
          )}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="flex flex-col items-center justify-center space-y-2">
            <Upload className={cn("text-muted-foreground", compact ? "h-6 w-6" : "h-10 w-10")} />
            <div className="space-y-1">
              <p className={cn("font-medium", compact ? "text-xs" : "text-sm")}>Drag and drop or click to upload</p>
              {!compact && <p className="text-xs text-muted-foreground">Supports JPG, PNG and GIF files</p>}
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
          </div>
        </div>
      ) : (
        <div className="relative border rounded-lg overflow-hidden bg-muted/10 transition-all duration-200 hover:shadow-md">
          <div className="flex items-center">
            <div className="relative w-24 h-24 flex-shrink-0">
              <img
                src={uploadedImage || "/placeholder.svg"}
                alt="Uploaded interior"
                className="w-full h-full object-cover"
              />
            </div>
            <div className="p-3 flex-1">
              <p className="text-sm font-medium truncate">Interior image uploaded</p>
              <p className="text-xs text-muted-foreground">Ready for transformation</p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-1 right-1 h-6 w-6 rounded-full bg-background/80 hover:bg-background"
              onClick={handleRemoveImage}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

