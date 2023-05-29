import axios from 'axios'

const baseUrl = 'http://localhost:8000/'

const predict = (formData) => {
  return axios.post(`${baseUrl}predict`, formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  }).then(response => response.data)
}

const aiService = { predict }

export default aiService