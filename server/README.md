# 远程执行服务端示例（remote_http）

本目录提供一个 **FastAPI** 服务端示例，用于配合客户端的 `execution.mode: remote_http`。

## 启动
```bash
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 客户端配置
在 `config/config.yaml`：
```yaml
execution:
  mode: "remote_http"
  remote_http:
    endpoint: "http://<server-ip>:8000/api/run"
    token: ""
```

## 安全建议（必须看）
- 示例默认不做沙箱隔离；请在容器/隔离环境中运行；
- 增加鉴权、CPU/内存/时间限制、依赖白名单等。
