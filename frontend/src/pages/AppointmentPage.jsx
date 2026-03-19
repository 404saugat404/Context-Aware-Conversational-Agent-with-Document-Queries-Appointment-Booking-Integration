import { useState } from 'react'
import { bookAppointment } from '../api/client'
import './AppointmentPage.css'

function AppointmentPage() {
  const [form, setForm] = useState({
    name: '',
    phone: '',
    email: '',
    preferred_date: '',
    reason: '',
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const data = await bookAppointment({
        ...form,
        reason: form.reason || undefined,
      })
      setResult(data)
      if (data.success) {
        setForm({ name: '', phone: '', email: '', preferred_date: '', reason: '' })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const today = new Date().toISOString().split('T')[0]

  return (
    <div className="appointment-page">
      <div className="appointment-container">
        <div className="appointment-header">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          <div>
            <h2>Book an Appointment</h2>
            <p>Fill in your details to schedule a visit.</p>
          </div>
        </div>

        <form className="appointment-form" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Full Name</label>
            <input
              id="name"
              name="name"
              type="text"
              required
              minLength={2}
              maxLength={100}
              value={form.name}
              onChange={handleChange}
              placeholder="John Doe"
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="phone">Phone Number</label>
              <input
                id="phone"
                name="phone"
                type="tel"
                required
                value={form.phone}
                onChange={handleChange}
                placeholder="+977 98XXXXXXXX"
              />
              <span className="form-hint">Nepali number (98/97...)</span>
            </div>

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                required
                value={form.email}
                onChange={handleChange}
                placeholder="john@example.com"
              />
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="preferred_date">Preferred Date</label>
            <input
              id="preferred_date"
              name="preferred_date"
              type="date"
              required
              min={today}
              value={form.preferred_date}
              onChange={handleChange}
            />
          </div>

          <div className="form-group">
            <label htmlFor="reason">Reason (optional)</label>
            <textarea
              id="reason"
              name="reason"
              maxLength={500}
              value={form.reason}
              onChange={handleChange}
              placeholder="Brief description of your visit reason..."
              rows={3}
            />
          </div>

          <button className="btn-book" type="submit" disabled={loading}>
            {loading ? 'Booking...' : 'Book Appointment'}
          </button>
        </form>

        {result && (
          <div className={`result-card ${result.success ? 'success' : 'fail'}`}>
            <p className="result-message">{result.message}</p>
            {result.appointment_id && (
              <p className="result-id">Appointment ID: <strong>{result.appointment_id}</strong></p>
            )}
          </div>
        )}

        {error && (
          <div className="result-card fail">
            <p className="result-message">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default AppointmentPage
