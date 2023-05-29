import { useState } from "react"
import ImageConverter from './ImageConverter'

const UploadAndDisplayImage = () => {
  const [selectedImage, setSelectedImage] = useState(null)

  const handleRemoveImage = () => {
    setSelectedImage(null)
    document.querySelector('input[type="file"]').value = null
  }

  const handleUploadImage = ( event ) => {
    event.preventDefault()
    if (!event.target.files[0]) {
      return null
    }
    const file = event.target.files[0];
          if (selectedImage && selectedImage.name === file.name) {
            setSelectedImage(selectedImage)
          } else {
            setSelectedImage(file)
          }
  }

  return (
    <div>
      <h1>Upload image</h1>
      {selectedImage && (
      <div>
        <img
          alt="not found"
          width={"200px"}
          src={URL.createObjectURL(selectedImage)}
          />
          <br />
          <button onClick={handleRemoveImage}>Remove</button>
      </div>
      )}
      <br />
      <br />
      <input
        type="file"
        name="myImage"
        onChange={(event) => handleUploadImage(event)}
      />
      <br />
      <br />
      {selectedImage && (
        <ImageConverter imageToConvert={selectedImage} />
      )}
    </div>
  )
}

export default UploadAndDisplayImage