#!/usr/bin/env python3
"""
Huggingface 모델을 GGUF로 변환하여 로컬에 저장하는 CLI 도구
Ollama 의존성 없음 - 순수 변환만 수행
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

# 설정
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
GGUF_DIR = MODELS_DIR / "gguf"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"
CONVERT_TIMEOUT = int(os.environ.get("CONVERT_TIMEOUT", "1800"))  # 기본 30분

# 디렉토리 생성
GGUF_DIR.mkdir(parents=True, exist_ok=True)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    """로그 출력"""
    print(f"[converter] {msg}", flush=True)


def detect_model_type(model_id: str) -> str:
    """모델 타입 감지 (gguf, safetensors, pytorch)"""
    api = HfApi()
    try:
        files = api.list_repo_files(model_id)
        
        # GGUF 파일 확인
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if gguf_files:
            return "gguf"
        
        # Safetensors 확인
        if any(f.endswith('.safetensors') for f in files):
            return "safetensors"
        
        # PyTorch 확인
        if any(f.endswith('.bin') or f.endswith('.pt') for f in files):
            return "pytorch"
        
        return "unknown"
    except Exception as e:
        log(f"모델 타입 감지 실패: {e}")
        return "unknown"


def get_best_gguf_file(model_id: str, quantization: str = "q4_k_m") -> Optional[str]:
    """가장 적합한 GGUF 파일 선택"""
    api = HfApi()
    files = api.list_repo_files(model_id)
    gguf_files = [f for f in files if f.endswith('.gguf')]
    
    if not gguf_files:
        return None
    
    # 선호하는 양자화 타입 순서
    preferred = [quantization, "q4_k_m", "q4_k_s", "q4_0", "q5_k_m", "q8_0"]
    
    for quant in preferred:
        for f in gguf_files:
            if quant in f.lower():
                return f
    
    # 없으면 첫 번째 파일 선택
    return gguf_files[0]


def download_gguf(model_id: str, model_name: str, quantization: str = "q4_k_m") -> Optional[Path]:
    """GGUF 모델 다운로드"""
    log(f"GGUF 모델 다운로드 중: {model_id}")
    
    gguf_file = get_best_gguf_file(model_id, quantization)
    if not gguf_file:
        log("GGUF 파일을 찾을 수 없습니다")
        return None
    
    log(f"선택된 파일: {gguf_file}")
    
    # 임시 경로에 다운로드
    temp_path = hf_hub_download(
        repo_id=model_id,
        filename=gguf_file,
        cache_dir=str(HF_CACHE_DIR),
    )
    
    # 모델별 폴더 생성 및 출력
    output_dir = MODELS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}.gguf"
    shutil.copy2(temp_path, output_path)
    log(f"저장 완료: {output_path}")
    
    return output_path


def download_safetensors(model_id: str) -> Optional[Path]:
    """Safetensors 모델 다운로드 (전체 repo)"""
    log(f"Safetensors 모델 다운로드 중: {model_id}")
    
    local_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(HF_CACHE_DIR),
        local_dir=str(HF_CACHE_DIR / model_id.replace("/", "_")),
        ignore_patterns=["*.md", "*.txt", "*.git*"]
    )
    
    return Path(local_path)


def convert_safetensors_to_gguf(model_path: Path, model_name: str, quantization: str = "q4_k_m") -> Optional[Path]:
    """Safetensors를 GGUF로 변환"""
    log(f"Safetensors -> GGUF 변환 중 (양자화: {quantization})")
    
    # F16 GGUF 생성 (임시)
    f16_output = GGUF_DIR / f"{model_name}-f16.gguf"
    
    try:
        # llama.cpp의 convert 스크립트 사용
        result = subprocess.run([
            "python3", "/app/llama.cpp/convert_hf_to_gguf.py",
            str(model_path),
            "--outfile", str(f16_output),
            "--outtype", "f16"
        ], capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
        
        if result.returncode != 0:
            log(f"F16 변환 실패: {result.stderr}")
            return None
        
        log(f"F16 GGUF 생성: {f16_output}")
        
        # 모델별 폴더 생성
        output_dir = MODELS_DIR / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 양자화 (f16이 아닌 경우)
        if quantization != "f16":
            quantized_output = output_dir / f"{model_name}.gguf"
            
            result = subprocess.run([
                "/app/llama.cpp/build/bin/llama-quantize",
                str(f16_output),
                str(quantized_output),
                quantization
            ], capture_output=True, text=True, timeout=CONVERT_TIMEOUT)
            
            if result.returncode != 0:
                log(f"양자화 실패: {result.stderr}")
                # F16 파일을 최종 출력으로 이동
                final_output = output_dir / f"{model_name}.gguf"
                shutil.move(str(f16_output), str(final_output))
                return final_output
            
            log(f"양자화 완료: {quantized_output}")
            
            # F16 파일 삭제 (용량 절약)
            f16_output.unlink()
            return quantized_output
        else:
            # F16을 최종 출력으로 이동
            final_output = output_dir / f"{model_name}.gguf"
            shutil.move(str(f16_output), str(final_output))
            return final_output
        
    except subprocess.TimeoutExpired:
        log("변환 시간 초과")
        return None
    except Exception as e:
        log(f"변환 오류: {e}")
        return None


def detect_model_family(model_id: str) -> str:
    """모델 패밀리 감지 (qwen, llama, mistral 등)"""
    model_lower = model_id.lower()
    
    if "qwen" in model_lower:
        return "qwen"
    elif "llama" in model_lower:
        return "llama"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    elif "phi" in model_lower:
        return "phi"
    elif "gemma" in model_lower:
        return "gemma"
    elif "nomic" in model_lower or "embed" in model_lower:
        return "embedding"
    else:
        return "generic"


def create_modelfile(gguf_path: Path, model_name: str, model_id: str = "") -> Path:
    """Ollama Modelfile 생성 (모델 패밀리별 최적화)"""
    modelfile_path = gguf_path.parent / f"Modelfile.{model_name}"
    
    # GGUF 파일명만 추출 (상대 경로로 사용)
    gguf_filename = gguf_path.name
    
    # 모델 패밀리 감지
    family = detect_model_family(model_id or model_name)
    log(f"모델 패밀리: {family}")
    
    # 패밀리별 템플릿
    if family == "qwen":
        # Qwen2.5 - Tool Calling 지원
        template = '''TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- if .Tools }}<|im_start|>user
# Tools

You have access to the following tools:

{{- range .Tools }}
{{ . }}
{{- end }}

# Instructions

You are a helpful assistant. When a tool is needed, respond with a JSON object in this format:
<tool_call>
{"name": "function_name", "arguments": {"arg1": "value1"}}
</tool_call>
<|im_end|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<tool_call>"
PARAMETER num_ctx 8192'''

    elif family == "llama":
        # Llama 3.x - Tool Calling 지원
        template = '''TEMPLATE """<|begin_of_text|>
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}
{{- if .Tools }}<|start_header_id|>system<|end_header_id|>

You have access to the following tools:
{{- range .Tools }}
{{ . }}
{{- end }}

When you need to use a tool, respond with JSON: {"name": "tool_name", "arguments": {...}}<|eot_id|>{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|}

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

"""

PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|end_of_text|>"
PARAMETER num_ctx 8192'''

    elif family == "mistral":
        # Mistral/Mixtral - Tool Calling 지원
        template = '''TEMPLATE """[INST] 
{{- if .System }}{{ .System }}

{{ end }}
{{- if .Tools }}You have access to these tools:
{{- range .Tools }}
{{ . }}
{{- end }}

Respond with JSON when using tools: {"name": "tool_name", "arguments": {...}}

{{ end }}{{ .Prompt }} [/INST]
{{- if .Response }} {{ .Response }}</s>{{ end }}"""

PARAMETER stop "[INST]"
PARAMETER stop "</s>"
PARAMETER num_ctx 8192'''

    elif family == "phi":
        # Microsoft Phi
        template = '''TEMPLATE """<|system|>
{{ if .System }}{{ .System }}{{ else }}You are a helpful assistant.{{ end }}<|end|>
<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
"""

PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER num_ctx 4096'''

    elif family == "gemma":
        # Google Gemma
        template = '''TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""

PARAMETER stop "<end_of_turn>"
PARAMETER num_ctx 8192'''

    elif family == "embedding":
        # Embedding 모델 (템플릿 불필요)
        template = '''# Embedding model - no template needed
PARAMETER num_ctx 2048'''

    else:
        # Generic ChatML 형식
        template = '''TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096'''

    content = f'''# Ollama Modelfile for {model_name}
# Model family: {family}
# Source: {model_id}
# 사용법: ollama create {model_name} -f Modelfile.{model_name}

FROM ./{gguf_filename}

{template}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
'''
    
    modelfile_path.write_text(content)
    log(f"Modelfile 생성: {modelfile_path}")
    
    return modelfile_path


def convert_model(model_id: str, model_name: str = None, quantization: str = "q4_k_m") -> bool:
    """메인 변환 함수"""
    
    # 모델 이름 자동 생성
    if not model_name:
        # "Qwen/Qwen2.5-0.5B-Instruct-GGUF" -> "qwen2.5-0.5b-instruct"
        name_part = model_id.split("/")[-1]
        name_part = name_part.replace("-GGUF", "").replace("-gguf", "")
        model_name = name_part.lower()
    
    log(f"변환 시작: {model_id} -> {model_name}")
    
    try:
        # 모델 타입 감지
        model_type = detect_model_type(model_id)
        log(f"모델 타입: {model_type}")
        
        if model_type == "gguf":
            # GGUF 직접 다운로드
            gguf_path = download_gguf(model_id, model_name, quantization)
            if not gguf_path:
                log("GGUF 다운로드 실패")
                return False
                
        elif model_type == "safetensors":
            # Safetensors 다운로드 및 변환
            model_path = download_safetensors(model_id)
            if not model_path:
                log("Safetensors 다운로드 실패")
                return False
            
            gguf_path = convert_safetensors_to_gguf(model_path, model_name, quantization)
            if not gguf_path:
                log("GGUF 변환 실패")
                return False
                
        else:
            log(f"지원되지 않는 모델 타입: {model_type}")
            return False
        
        # Modelfile 생성 (model_id 전달하여 패밀리 감지)
        modelfile_path = create_modelfile(gguf_path, model_name, model_id)
        
        log(f"✓ 변환 완료: {gguf_path}")
        log(f"  파일 크기: {gguf_path.stat().st_size / 1024 / 1024:.1f} MB")
        log(f"  Modelfile: {modelfile_path}")
        log(f"")
        log(f"Ollama에 등록하려면:")
        log(f"  cd {gguf_path.parent}")
        log(f"  ollama create {model_name} -f Modelfile.{model_name}")
        return True
        
    except Exception as e:
        log(f"변환 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """CLI 진입점"""
    if len(sys.argv) < 3:
        print("사용법: python converter.py convert <model_id> [model_name] [quantization]")
        print("")
        print("예시:")
        print("  python converter.py convert Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen q4_k_m")
        print("  python converter.py convert microsoft/phi-2 phi2 q4_k_m")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "convert":
        model_id = sys.argv[2]
        model_name = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
        quantization = sys.argv[4] if len(sys.argv) > 4 else "q4_k_m"
        
        success = convert_model(model_id, model_name, quantization)
        sys.exit(0 if success else 1)
    else:
        print(f"알 수 없는 명령: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
