# Library and Information Science Chatbot
## pdf 문서 기반 챗봇 만들기 이전 가상환경 작업

가상환경 생성하기
VSC 상단의 'Terminal'을 클릭하고, 'New Terminal'을 누르면 터미널을 열 수 있습니다. 이 때, 터미널이 열리는 경로를 확인하세요. 터미널의 경로는 streamlit-tutorial 폴더로 지정이 되어 있어야 합니다. 해당 터미널의 입력창에서 다음과 같이 코드를 순서대로 입력합니다.

이 튜토리얼에서 streamlit을 실행하기 위해 가상환경을 실행합니다. venv는 파이썬의 대표적인 가상환경 라이브러리로, 프로젝트마다 다른 버전의 패키지를 사용하고 싶을 때 개별 프로젝트를 실행할 수 있는 격리된 환경을 제공합니다.

# 1. generate python bubble
python -m venv env

# 2. activate
source env/bin/activate  # mac
env\Scripts\activate.bat  # window

# 3. install dependencies
pip install -r requirements.txt

# 4. deactivate bubble
# deactivate
streamlit 실행은 가상환경에 들어와있는 상태로 진행해야 합니다. 위의 코드에서 3번까지 진행하고, Streamlit을 실행하면 됩니다. 4번의 코드는 가상환경을 나갈 때 사용합니다.

Streamlit 실행하기
위의 가상환경이 실행된 상태에서 다음의 코드를 터미널에 작성합니다. 정상적으로 streamlit이 실행된다면 localhost:8501에서 튜토리얼을 확인할 수 있습니다.

streamlit run main.py
새로운 파일을 Streamlit으로 열고 싶다면, 기존에 실행하고 있던 streamlit을 중단하고(ctrl + c) 같은 경로에서 다음을 입력합니다.

# 새로운 파일 이름이 test.py인 경우
streamlit run test.py
