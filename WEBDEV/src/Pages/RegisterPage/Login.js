import React, { useEffect, useState, useContext } from 'react';
import { FiMail } from "react-icons/fi";
import { RiLockPasswordLine } from "react-icons/ri";
import "../RegisterPage/RegisterPage.css";
import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { AuthContext } from '../../context/AuthContext'; // Import AuthContext

const Login = ({name,setname}) => {
    const navigate = useNavigate();
    const { setAuth } = useContext(AuthContext); // Access setAuth from context
    const [error, setError] = useState({});
    const [submit, setSubmit] = useState(false);
  
    const [data, setData] = useState({
        email: "",
        password: "",
    });

    const handleChange = (e) => {
        const newObj = { ...data, [e.target.name]: e.target.value };
        setData(newObj);
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        const validationErrors = validationLogin(data);
        setError(validationErrors);
        setSubmit(true);

        if (Object.keys(validationErrors).length === 0) {
            try {
                const response = await axios.post('http://localhost:5000/login', data);
                if (response.status === 200) {
                    const token = response.data.token;
                    setname(response.data.full_name);
                    const expiryTime = new Date().getTime() + 60 * 60 * 1000; // 1 hour from now

                    // Store token and expiry time in localStorage
                    localStorage.setItem('token', token);
                    localStorage.setItem('tokenExpiry', expiryTime);

                    // Update auth state in context
                    setAuth({
                        token: token,
                        isAuthenticated: true,
                    });

                    navigate("/home");
                }
            } catch (err) {
                if (err.response && err.response.data && err.response.data.message) {
                    setError({ server: err.response.data.message });
                } else {
                    setError({ server: 'An error occurred. Please try again.' });
                }
            }
        }
    };

    useEffect(() => {
        // No need to handle submit here since it's managed in handleLogin
    }, [error]);

    function validationLogin(data){
        const error = {};

        const emailPattern= /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}$/;
        const passwordPattern= /^[a-zA-Z0-9!@#\$%\^\&*_=+-]{8,12}$/g;

        if(data.email === ""){
            error.email ="* Email is Required";
        }
        else if(!emailPattern.test(data.email)){
            error.email="* Email did not match";
        }

        if(data.password === ""){
            error.password = "* Password is Required";
        }
        else if(!passwordPattern.test(data.password)){
            error.password="* Password not valid";
        }
        
        return error;
    }

    return (
        <div className="container">
            <div className="container-form">
                <form onSubmit={handleLogin}>
                    <h1>Login</h1>
                    <p>Please sign in to continue.</p>

                    {error.server && <span style={{color:"red", display:"block", marginTop:"5px"}}>{error.server}</span>}

                    <div className="inputBox">
                        <FiMail className='mail'/>
                        <input type="email" 
                                name="email" 
                                id="email" 
                                onChange={handleChange}
                                placeholder='Email'/> 
                    </div>
                    {error.email && <span style={{color:"red", display:"block", marginTop:"5px"}}>{error.email}</span>}

                    <div className="inputBox">
                        <RiLockPasswordLine className='password'/>
                        <input type="password" 
                                name="password" 
                                id="password" 
                                onChange={handleChange}
                                placeholder='Password'/>
                    </div>
                    {error.password && <span style={{color:"red", display:"block", marginTop:"5px"}}>{error.password}</span>}

                    <div className='divBtn'>
                        <small className='FG'>Forgot Password?</small>
                        <button type='submit' className='loginBtn'>LOGIN</button>
                    </div>
                    
                </form>

                <div className='dont'>
                    <p>Don't have an account? <Link to="/signup"><span>Sign up</span></Link></p>
                </div>
            </div>
        </div>
    );
};

export default Login;
