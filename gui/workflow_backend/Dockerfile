FROM python:3.11.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetryをインストール
RUN pip install poetry==1.7.1

# Poetryの設定（仮想環境を作成しない）
RUN poetry config virtualenvs.create false

# 依存関係ファイルをコピー
COPY pyproject.toml poetry.lock* ./

# 依存関係をインストール
RUN poetry install --no-interaction --no-ansi --no-root

# プロジェクトファイルをコピー
COPY . .

# 8000番ポートを公開
EXPOSE 3000

# コマンドを実行
CMD ["python", "django-project/manage.py", "runserver", "0.0.0.0:3000"]
