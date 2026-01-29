#!/bin/bash
# Huggingface 모델을 GGUF로 변환하여 로컬에 저장하는 일회성 작업 컨테이너
# Ollama 의존성 없음 - 순수 변환만 수행

set -e

echo "=========================================="
echo "  Huggingface -> GGUF 모델 변환"
echo "=========================================="

# 환경변수 확인
if [ -z "$AUTO_CONVERT_MODELS" ]; then
    echo "오류: AUTO_CONVERT_MODELS 환경변수가 설정되지 않았습니다."
    echo ""
    echo "사용법:"
    echo "  AUTO_CONVERT_MODELS=\"huggingface_model_id\""
    echo "  OUTPUT_MODEL_NAME=\"output_name\" (선택)"
    echo "  QUANTIZATION=\"q4_k_m\" (선택, 기본값: q4_k_m)"
    echo ""
    echo "예시:"
    echo "  AUTO_CONVERT_MODELS=Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    echo "  OUTPUT_MODEL_NAME=qwen2.5"
    echo "  QUANTIZATION=q4_k_m"
    exit 1
fi

MODEL_ID="$AUTO_CONVERT_MODELS"
OUTPUT_NAME="${OUTPUT_MODEL_NAME:-}"
QUANT="${QUANTIZATION:-q4_k_m}"

echo ""
echo "설정:"
echo "  모델 ID: $MODEL_ID"
echo "  출력 이름: ${OUTPUT_NAME:-auto}"
echo "  양자화: $QUANT"
echo "  출력 경로: /models"
echo ""

# 모델 변환 시작
echo "=========================================="
echo "  변환 시작"
echo "=========================================="

# Python 변환 스크립트 호출
if [ -n "$OUTPUT_NAME" ]; then
    python3 /app/converter.py convert "$MODEL_ID" "$OUTPUT_NAME" "$QUANT"
else
    python3 /app/converter.py convert "$MODEL_ID" "" "$QUANT"
fi

RESULT=$?

# 결과 출력
echo ""
echo "=========================================="
echo "  변환 완료"
echo "=========================================="

if [ $RESULT -eq 0 ]; then
    echo "✓ 성공!"
    echo ""
    echo "생성된 파일:"
    find /models -name "*.gguf" -type f 2>/dev/null | while read f; do
        size=$(du -h "$f" | cut -f1)
        echo "  - $f ($size)"
    done
else
    echo "✗ 실패"
fi

echo ""
echo "=========================================="

exit $RESULT
