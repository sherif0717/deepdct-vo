for sequence in 09 10; do
  if [[ "${sequence}" == "09" ]]; then
    translation_scale=0.975
  else
    translation_scale=1.007
  fi

  python scripts/evaluate_deepdct_vo.py \
    --checkpoint "${A7_CHECKPOINT}" \
    --data-root "${A7_DATA_ROOT}" \
    --sequence "${sequence}" \
    --require-a6-protocol \
    --device "${A7_DEVICE:-auto}" \
    --model-t-rotation-source ground_truth \
    --trajectory-rotation-source ground_truth \
    --translation-scale-factor "${translation_scale}" \
    --output-dir \
      "${A7_OUTPUT_ROOT}/evaluation_sequence_${sequence}_gt_gt"
done
