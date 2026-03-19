const BASE = '/api/v1'

export async function sendChat(message, sessionId, model) {
  const body = { message }
  if (sessionId) body.session_id = sessionId
  if (model) body.model = model

  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Chat request failed')
  }
  return res.json()
}

export async function bookAppointment(data) {
  const res = await fetch(`${BASE}/appointment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Appointment booking failed')
  }
  return res.json()
}

export async function getModels() {
  const res = await fetch(`${BASE}/models`)
  if (!res.ok) throw new Error('Failed to fetch models')
  return res.json()
}

export async function deleteSession(sessionId) {
  const res = await fetch(`${BASE}/session/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to clear session')
  return res.json()
}
