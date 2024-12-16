
# Computer_Architecture_HW

This repository contains scripts and outputs related to VLLM service performance testing and usability validation.

## Files

- **start_server.sh**: Shell script to start the VLLM service.
- **hf_test.py**: Python script for testing performance using method 1.
- **vllm_test.py**: Python script for testing performance using method 2.
- **test.sh**: Shell script to verify VLLM service availability.
- **inference_stats.txt**: Output file containing inference performance statistics.
- **vlm_inference_stats.txt**: Output file with additional VLLM inference statistics.

## Usage

1. **Start the VLLM Service**:
   ```bash
   bash start_server.sh
   ```

2. **Run Performance Tests**:
   - Method 1:
     ```bash
     python hf_test.py
     ```
   - Method 2:
     ```bash
     python vllm_test.py
     ```

3. **Check VLLM Usability**:
   ```bash
   bash test.sh
   ```

4. **Outputs**:
   - `inference_stats.txt` and `vlm_inference_stats.txt` contain performance results.

---

## Prerequisites

- Python 3.x
- vLLM 0.6.3.post1

---

## Author

- **Zh1yuShen**
