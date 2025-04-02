"use client"

import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import type { ReactNode } from "react"

interface PromptInputProps {
  value: string
  onChange: (value: string) => void
  disabled?: boolean
  placeholder?: string
  hint?: ReactNode
}

export function PromptInput({
  value,
  onChange,
  disabled,
  placeholder = "Describe how you want to transform the image...",
  hint,
}: PromptInputProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="prompt" className="text-sm font-medium">
        Design Prompt
      </Label>
      <Textarea
        id="prompt"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className="min-h-[100px] resize-none transition-colors"
      />
      {hint}
    </div>
  )
}

