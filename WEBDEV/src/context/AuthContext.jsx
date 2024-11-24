// src/context/AuthContext.js

import React, { createContext, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const navigate = useNavigate();
    const [auth, setAuth] = useState({
        token: null,
        isAuthenticated: false,
    });

    useEffect(() => {
        // Retrieve token and expiry from localStorage
        const token = localStorage.getItem('token');
        const tokenExpiry = localStorage.getItem('tokenExpiry');

        if (token && tokenExpiry) {
            const currentTime = new Date().getTime();

            if (currentTime < tokenExpiry) {
                // Token is still valid
                setAuth({
                    token: token,
                    isAuthenticated: true,
                });
            } else {
                // Token expired
                localStorage.removeItem('token');
                localStorage.removeItem('tokenExpiry');
                setAuth({
                    token: null,
                    isAuthenticated: false,
                });
                navigate("/");
            }
        }
    }, [navigate]);

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('tokenExpiry');
        setAuth({
            token: null,
            isAuthenticated: false,
        });
        navigate("/");
    };

    return (
        <AuthContext.Provider value={{ auth, setAuth, logout }}>
            {children}
        </AuthContext.Provider>
    );
};
