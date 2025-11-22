import axios from 'axios';


 export const URL = "http://localhost:3000";
export const PY_URL = "http://localhost:5000";


const api = axios.create({
  baseURL: API_BASE_URL,
});


api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
