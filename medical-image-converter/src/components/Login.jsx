import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, currentUser } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (currentUser) {
      navigate('/');
    }
  }, [currentUser, navigate]);

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      setError('');
      setLoading(true);
      await login(email, password);
      navigate('/');
    } catch (error) {
      setError('Failed to log in: ' + error.message);
    }
    setLoading(false);
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-950 text-white relative px-4">
      <div className="w-full max-w-md p-8 space-y-8 bg-gray-800/70 backdrop-blur-md border border-gray-700 rounded-lg shadow-lg animate-fade-in">
        <h2 className="text-3xl font-extrabold text-center bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 text-transparent bg-clip-text animate-text-glow">
          Sign in to your account
        </h2>

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500 text-red-300 text-sm rounded">
            {error}
          </div>
        )}

        <form className="space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300">
                Email address
              </label>
              <input
                type="email"
                required
                className="w-full p-2 mt-1 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300">
                Password
              </label>
              <input
                type="password"
                required
                className="w-full p-2 mt-1 bg-gray-700 border border-gray-600 rounded focus:ring-2 focus:ring-blue-500"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <div className="flex items-center justify-end">
            <Link to="/forgot-password" className="text-sm text-blue-400 hover:text-blue-300">
              Forgot your password?
            </Link>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex justify-center py-2 px-4 rounded-md shadow-md text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <div className="text-center">
          <p className="text-sm text-gray-400">
            Need an account?{' '}
            <Link to="/signup" className="text-blue-400 hover:text-blue-300">
              Sign up
            </Link>
          </p>
        </div>
      </div>

      {/* Animation styles */}
      <style>{`
        @keyframes textGlow {
          0%, 100% { text-shadow: 0 0 10px #3b82f6, 0 0 20px #06b6d4; }
          50% { text-shadow: 0 0 20px #8b5cf6, 0 0 30px #06b6d4; }
        }
        .animate-text-glow {
          animation: textGlow 3s ease-in-out infinite alternate;
        }
        .animate-fade-in {
          animation: fadeIn 1s ease-out both;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
