ps -ef | grep OCR | grep -v grep | cut -c 9-15 | xargs kill -s 9
nohup  python -u /home/zhaozhiwwei/OCR/sever.py > /home/zhaozhiwei/OCR/ocr.nohup.out 2>&1 &