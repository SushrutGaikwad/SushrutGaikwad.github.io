--[[
  cleveref_capitalize.lua

  Mimics cleveref's \Cref vs \cref for Quarto cross-references.

  Quarto renders a reference like `@alg-foo` with a fixed per-type prefix
  (e.g. lowercase "alg. 2", but capitalized "Figure 3"). It does NOT detect
  sentence position. This filter runs AFTER Quarto resolves cross-references
  (scheduled with `at: post-quarto`), at which point each reference is a Link
  with class "quarto-xref" and text such as "alg.\u{A0}2". It capitalizes the
  first letter of that text when, and only when, the reference sits at the
  start of a sentence. Capitalizing an already-capital prefix ("Figure") is a
  no-op, and references with no leading letter ("(1)") are left untouched.
]]

-- Does this string end a sentence (allowing trailing closing quotes/brackets)?
local function ends_sentence(text)
  return text:match("[%.%!%?%:][\"'%)%]]*$") ~= nil
end

-- Is this inline a resolved Quarto cross-reference?
local function is_xref(inline)
  if inline.t ~= "Link" then return false end
  for _, cls in ipairs(inline.classes) do
    if cls == "quarto-xref" then return true end
  end
  return false
end

-- Capitalize the first alphabetic character in the first Str of an inline list.
local function capitalize_first_str(inlines)
  for _, inl in ipairs(inlines) do
    if inl.t == "Str" then
      local pre, ch, rest = inl.text:match("^(%A*)(%a)(.*)$")
      if ch then
        inl.text = pre .. ch:upper() .. rest
      end
      return
    elseif inl.t ~= "Space" and inl.t ~= "SoftBreak" and inl.t ~= "LineBreak" then
      return
    end
  end
end

-- Walk a single paragraph's inlines, tracking sentence boundaries.
local function process(inlines)
  local at_start = true
  for _, inl in ipairs(inlines) do
    local t = inl.t
    if t == "Space" or t == "SoftBreak" or t == "LineBreak" then
      -- whitespace carries the current sentence state forward
    elseif t == "Str" then
      at_start = ends_sentence(inl.text)
    else
      if at_start and is_xref(inl) then
        capitalize_first_str(inl.content)
      end
      at_start = false
    end
  end
end

function Para(el)
  process(el.content)
  return el
end

function Plain(el)
  process(el.content)
  return el
end
