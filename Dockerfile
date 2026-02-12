FROM python:3.12

WORKDIR /app

# uv 설치
RUN pip install uv



# 컨테이너 실행 시 유지
CMD ["/bin/bash"]
