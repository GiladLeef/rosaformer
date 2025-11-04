"""
================================================================================
# Copyright (c) 2025 Gilad Leef
#
# This software is provided for educational, research, and personal use only.
# Commercial use, resale, or distribution for profit is strictly prohibited.
# All modifications and derivative works must be distributed under the same license terms.
#
# Any disputes arising from the use of this software shall be governed by and construed in accordance with the laws of the State of Israel.
# Exclusive jurisdiction for any such disputes shall lie with the competent courts located in Israel.
================================================================================
"""

import torch
from transformers import PreTrainedTokenizerFast, TextIteratorStreamer
from pathlib import Path
import sys
from threading import Thread

import model
from config import applyConfig, getMinipileConfig


"""
================================================================================
Configuration
================================================================================
"""

outputDir = "./output/minipile"
tokenizerPath = "./tokenizer/minipile.json"

maxNewTokens = 64
temperature = 0.8
topP = 0.9
topK = 50
repetitionPenalty = 1.1


def findLatestCheckpoint(outputDir: str):
    """Find the latest checkpoint in output directory."""
    outputPath = Path(outputDir)
    
    if not outputPath.exists():
        return None
    
    checkpoints = []
    
    for item in outputPath.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            if item.name == "checkpoint-interrupted":
                checkpoints.append((float('inf'), item))
            else:
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append((step, item))
                except ValueError:
                    continue
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return str(checkpoints[0][1])


"""
================================================================================
Chat Functions
================================================================================
"""

def loadModel(checkpointPath: str = None):
    """Load trained model from checkpoint."""
    if checkpointPath is None:
        checkpointPath = findLatestCheckpoint(outputDir)
        if checkpointPath is None:
            print(f"Error: No checkpoints found in {outputDir}")
            print("Please train the model first using: python train/minipile.py")
            sys.exit(1)
    
    print(f"Loading model from checkpoint: {checkpointPath}")
    
    applyConfig(model, getMinipileConfig())
    
    try:
        chatModel = model.RosaformerForCausalLM.from_pretrained(checkpointPath)
        print("  Model loaded using from_pretrained()")
    except Exception as e:
        print(f"  from_pretrained() failed: {e}")
        print("  Trying manual loading...")
        
        config = model.createRosaformerConfig()
        chatModel = model.RosaformerForCausalLM(config)
        
        checkpointFile = Path(checkpointPath) / "pytorch_model.bin"
        if not checkpointFile.exists():
            checkpointFile = Path(checkpointPath) / "model.safetensors"
            if not checkpointFile.exists():
                print(f"Error: No model checkpoint found at {checkpointPath}")
                print("Available files:", list(Path(checkpointPath).iterdir()))
                sys.exit(1)
        
        stateDict = torch.load(checkpointFile, map_location="cpu", weights_only=False)
        chatModel.load_state_dict(stateDict)
        print("  Model loaded manually")
    
    if torch.cuda.is_available():
        chatModel = chatModel.cuda()
        print("  Model moved to GPU")
    else:
        print("  Model running on CPU")
    
    chatModel.eval()
    return chatModel


def loadTokenizer(tokenizerPath: str):
    """Load tokenizer."""
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizerPath)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '</s>'})
    return tokenizer


def generateStreaming(
    chatModel,
    tokenizer,
    prompt: str,
    maxNewTokens: int = 64,
    temperature: float = 0.8,
    topP: float = 0.9,
    topK: int = 50,
    repetitionPenalty: float = 1.1
):
    """Generate response from prompt with streaming output."""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generationKwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=maxNewTokens,
        temperature=temperature,
        top_p=topP,
        top_k=topK,
        repetition_penalty=repetitionPenalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    thread = Thread(target=chatModel.generate, kwargs=generationKwargs)
    thread.start()
    
    generatedText = ""
    for newText in streamer:
        generatedText += newText
        print(newText, end="", flush=True)
    
    thread.join()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generatedText


def chat():
    """Main chat loop."""
    print("="*80)
    print("ROSA Model Chat Interface")
    print("="*80)
    print()
    
    chatModel = loadModel()
    tokenizer = loadTokenizer(tokenizerPath)
    
    print()
    print("="*80)
    print("Chat loaded! Type your message and press Enter.")
    print("Commands:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history")
    print("  /temp X   - Set temperature (0.1-2.0)")
    print("  /tokens X - Set max new tokens (1-256)")
    print("  /quit     - Exit chat")
    print("\nTip: Short responses (64 tokens) work best to avoid OOM")
    print("="*80)
    print()
    
    conversationHistory = []
    
    global maxNewTokens, temperature
    
    while True:
        try:
            userInput = input("\nYou: ").strip()
            
            if not userInput:
                continue
            
            if userInput.startswith("/"):
                command = userInput[1:].lower().split()
                
                if command[0] == "quit" or command[0] == "exit":
                    print("\nGoodbye!")
                    break
                
                elif command[0] == "help":
                    print("\nCommands:")
                    print("  /help     - Show this help message")
                    print("  /clear    - Clear conversation history")
                    print("  /temp X   - Set temperature (0.1-2.0)")
                    print("  /tokens X - Set max new tokens (1-1024)")
                    print("  /quit     - Exit chat")
                    continue
                
                elif command[0] == "clear":
                    conversationHistory = []
                    print("\nConversation history cleared!")
                    continue
                
                elif command[0] == "temp" or command[0] == "temperature":
                    if len(command) > 1:
                        try:
                            temperature = float(command[1])
                            temperature = max(0.1, min(2.0, temperature))
                            print(f"\nTemperature set to {temperature}")
                        except ValueError:
                            print("\nInvalid temperature value")
                    else:
                        print(f"\nCurrent temperature: {temperature}")
                    continue
                
                elif command[0] == "tokens":
                    if len(command) > 1:
                        try:
                            maxNewTokens = int(command[1])
                            maxNewTokens = max(1, min(256, maxNewTokens))
                            print(f"\nMax new tokens set to {maxNewTokens}")
                            if maxNewTokens > 128:
                                print("  Warning: High token counts may cause OOM")
                        except ValueError:
                            print("\nInvalid token count")
                    else:
                        print(f"\nCurrent max new tokens: {maxNewTokens}")
                    continue
                
                else:
                    print(f"\nUnknown command: {command[0]}")
                    print("Type /help for available commands")
                    continue
            
            conversationHistory.append(f"User: {userInput}")
            
            if len(conversationHistory) > 10:
                conversationHistory = conversationHistory[-10:]
            
            prompt = "\n".join(conversationHistory) + "\nAssistant:"
            
            print("\nAssistant: ", end="", flush=True)
            
            response = generateStreaming(
                chatModel,
                tokenizer,
                prompt,
                maxNewTokens=maxNewTokens,
                temperature=temperature,
                topP=topP,
                topK=topK,
                repetitionPenalty=repetitionPenalty
            )
            
            print()
            
            conversationHistory.append(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing chat...")


"""
================================================================================
Main Entry Point
================================================================================
"""

if __name__ == "__main__":
    chat()
