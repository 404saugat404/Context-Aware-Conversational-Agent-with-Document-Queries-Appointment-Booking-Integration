import { useState, useEffect, useRef } from 'react'
import { sendChat, getModels, deleteSession } from '../api/client'
import './ChatPage.css'

function ChatPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState('')
  const [ollamaStatus, setOllamaStatus] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    getModels()
      .then((data) => {
        const all = []
        if (data.gemini) data.gemini.forEach((m) => all.push({ name: m, provider: 'Gemini' }))
        if (data.ollama) data.ollama.forEach((m) => all.push({ name: m, provider: 'Ollama' }))
        setModels(all)
        setOllamaStatus(data.ollama_status)
        if (data.default_model) setSelectedModel(data.default_model)
        else if (all.length) setSelectedModel(all[0].name)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    setLoading(true)

    try {
      const data = await sendChat(text, sessionId, selectedModel)
      setSessionId(data.session_id)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.reply,
          intent: data.intent,
          sources: data.sources,
          model_used: data.model_used,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Error: ${err.message}`, isError: true },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewChat = async () => {
    if (sessionId) {
      try { await deleteSession(sessionId) } catch {}
    }
    setMessages([])
    setSessionId(null)
  }

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="chat-header-left">
          <h2>Chat</h2>
          {sessionId && (
            <span className="session-badge" title={sessionId}>
              Session active
            </span>
          )}
        </div>
        <div className="chat-header-right">
          <div className="model-select-wrapper">
            <label htmlFor="model-select">Model:</label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {m.name} ({m.provider})
                </option>
              ))}
            </select>
            {ollamaStatus && ollamaStatus !== 'connected' && (
              <span className="ollama-badge">Ollama offline</span>
            )}
          </div>
          <button className="btn-new-chat" onClick={handleNewChat}>
            New Chat
          </button>
        </div>
      </header>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h3>Start a conversation</h3>
            <p>Ask questions about documents, book appointments, or just say hello.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className={`bubble ${msg.role} ${msg.isError ? 'error' : ''}`}>
              <p>{msg.content}</p>
              {msg.role === 'assistant' && !msg.isError && (
                <div className="message-meta">
                  {msg.intent && <span className="meta-tag intent">{msg.intent}</span>}
                  {msg.model_used && <span className="meta-tag model">{msg.model_used}</span>}
                  {msg.sources && msg.sources.length > 0 && (
                    <span className="meta-tag sources">
                      Sources: {msg.sources.join(', ')}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="bubble assistant">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <div className="input-wrapper">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            rows={1}
            disabled={loading}
          />
          <button
            className="btn-send"
            onClick={handleSend}
            disabled={!input.trim() || loading}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
