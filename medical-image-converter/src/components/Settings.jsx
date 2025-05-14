import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { updatePassword, updateProfile } from 'firebase/auth';

const Settings = () => {
  const { currentUser, updateEmail } = useAuth();
  const [displayName, setDisplayName] = useState(currentUser?.displayName || '');
  const [email, setEmail] = useState(currentUser?.email || '');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleBack = () => {
    navigate('/');
  };

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');
    setLoading(true);

    try {
      const updates = [];
      
      // Update display name if changed
      if (displayName !== currentUser.displayName) {
        await updateProfile(currentUser, {
          displayName: displayName
        });
      }
      
      // Update email if changed
      if (email !== currentUser.email) {
        updates.push(updateEmail(email));
      }
      
      await Promise.all(updates);
      setMessage('Profile updated successfully');
    } catch (err) {
      setError('Failed to update profile: ' + err.message);
    }
    
    setLoading(false);
  };

  const handlePasswordUpdate = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');

    if (newPassword !== confirmPassword) {
      return setError('Passwords do not match');
    }

    if (newPassword.length < 6) {
      return setError('Password should be at least 6 characters');
    }

    setLoading(true);
    try {
      await updatePassword(currentUser, newPassword);
      setMessage('Password updated successfully');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      // Improve error message for auth/requires-recent-login
      if (err.code === 'auth/requires-recent-login') {
        setError('For security reasons, please log out and log back in before changing your password');
      } else {
        setError('Failed to update password: ' + err.message);
      }
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-2xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-bold">Settings</h1>
          <button 
            onClick={handleBack}
            className="bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded"
          >
            Back to Main Page
          </button>
        </div>
        
        {error && (
          <div className="bg-red-500 bg-opacity-20 border border-red-500 text-red-300 p-3 rounded mb-4">
            {error}
          </div>
        )}
        
        {message && (
          <div className="bg-green-500 bg-opacity-20 border border-green-500 text-green-300 p-3 rounded mb-4">
            {message}
          </div>
        )}

        {/* Profile Information */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Profile Information</h2>
          <form onSubmit={handleProfileUpdate}>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Display Name</label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500 text-white"
              />
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Email Address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500 text-white"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
            >
              Update Profile
            </button>
          </form>
        </div>

        {/* Change Password */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Change Password</h2>
          <form onSubmit={handlePasswordUpdate}>
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">New Password</label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500 text-white"
              />
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-2">Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="w-full p-2 bg-gray-700 rounded border border-gray-600 focus:ring-2 focus:ring-blue-500 text-white"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 disabled:opacity-50"
            >
              Change Password
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Settings; 