#!/bin/bash
# Check for inline prompts in non-prompts directories
# Exit with error code if any inline prompts are found

echo "🔍 Checking for inline prompts outside of src/prompts/..."

VIOLATIONS=""

# Check for inline system messages
SYSTEM_MSG_VIOLATIONS=$(grep -r 'system_message = """' src/ --exclude-dir=prompts 2>/dev/null | grep -v test || true)
if [ ! -z "$SYSTEM_MSG_VIOLATIONS" ]; then
    VIOLATIONS="${VIOLATIONS}${SYSTEM_MSG_VIOLATIONS}\n"
fi

# Check for inline prompt templates
PROMPT_VIOLATIONS=$(grep -r 'prompt = f"""' src/ --exclude-dir=prompts 2>/dev/null | grep -v test || true)
if [ ! -z "$PROMPT_VIOLATIONS" ]; then
    VIOLATIONS="${VIOLATIONS}${PROMPT_VIOLATIONS}\n"
fi

# Check for thinking_prompts dict
THINKING_VIOLATIONS=$(grep -r 'thinking_prompts = {' src/ --exclude-dir=prompts 2>/dev/null | grep -v test || true)
if [ ! -z "$THINKING_VIOLATIONS" ]; then
    VIOLATIONS="${VIOLATIONS}${THINKING_VIOLATIONS}\n"
fi

# Check for other common inline prompt patterns
OTHER_VIOLATIONS=$(grep -r 'prompt = """' src/ --exclude-dir=prompts 2>/dev/null | grep -v test || true)
if [ ! -z "$OTHER_VIOLATIONS" ]; then
    VIOLATIONS="${VIOLATIONS}${OTHER_VIOLATIONS}\n"
fi

# Check for triple-quoted blocks that might be prompts (be more lenient to avoid false positives)
TRIPLE_QUOTE_VIOLATIONS=$(grep -r '= """.*[Ss]ie sind.*' src/ --exclude-dir=prompts 2>/dev/null | grep -v test || true)
if [ ! -z "$TRIPLE_QUOTE_VIOLATIONS" ]; then
    VIOLATIONS="${VIOLATIONS}${TRIPLE_QUOTE_VIOLATIONS}\n"
fi

if [ ! -z "$VIOLATIONS" ]; then
    echo "❌ Found inline prompts that should be in YAML:"
    echo -e "$VIOLATIONS"
    echo ""
    echo "All prompts should be moved to src/prompts/configs/*.yaml files and accessed via get_prompt()"
    exit 1
fi

echo "✅ No inline prompts found - all prompts are properly centralized in YAML!"
exit 0