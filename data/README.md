# Data

Kaggle 대회 데이터는 라이선스·용량 문제로 커밋하지 않는다. 아래로 재현:

```bash
kaggle competitions download -c playground-series-s6e7 -p data
cd data && tar -xf playground-series-s6e7.zip
```

- `train.csv` 690,088행 (타겟 `health_condition` 포함)
- `test.csv` 295,753행
- `sample_submission.csv` 제출 양식
