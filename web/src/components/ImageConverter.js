import aiService from '../services/ai'
import { useState, useCallback } from 'react'

const ImageConverter = ( {imageToConvert} ) => {
  const [predictedWord, setPredictedWord] = useState(null);

  const convertImage = useCallback(async () => {
    const formData = new FormData()
    formData.append("image", imageToConvert)

    await aiService.predict(formData)
    .then((response) => {
      setPredictedWord(response)
    })
    .catch((err) => console.log(err))
  }, [imageToConvert])

  const handleConvertImage = () => {
    convertImage()
  }
  console.log()
  return (
    <div>
      <button onClick={handleConvertImage}>Convert</button>
      <br />
      {predictedWord && <img src={`data:image/jpeg;base64,${predictedWord.images[0]}`}/>}
      {predictedWord && <img src={`data:image/jpeg;base64,${predictedWord.images[1]}`}/>}
      {predictedWord && <div> <b>Recognized words:</b>< br /> {predictedWord.texts[0]}</div>}
      {predictedWord && <div>< br /> <b>After spelling correction:</b>< br /> {predictedWord.texts[1]}</div>}
      < br />< br />< br />< br />< br />< br />< br />< br />< br />< br />< br />< br />< br />< br />< br />
    </div>
  )
}

export default ImageConverter