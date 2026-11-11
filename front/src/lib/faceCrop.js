// Вырезание лица из видео: квадрат с запасом по краям -> 48x48 оттенки серого.
const SIZE = 48
const canvas = document.createElement('canvas')
canvas.width = canvas.height = SIZE
const ctx = canvas.getContext('2d', { willReadFrequently: true })
ctx.imageSmoothingEnabled = true
ctx.imageSmoothingQuality = 'high'

// bbox детектора покрывает брови–подбородок; в FER-2013 в кадре ещё и лоб,
// поэтому расширяем квадрат и немного смещаем вверх.
export function squareRegion(box, vw, vh) {
  const side = Math.min(Math.max(box.width, box.height) * 1.25, vw, vh)
  const cx = box.originX + box.width / 2
  const cy = box.originY + box.height / 2 - box.height * 0.08
  const sx = Math.min(Math.max(cx - side / 2, 0), vw - side)
  const sy = Math.min(Math.max(cy - side / 2, 0), vh - side)
  return { sx, sy, side }
}

/** Возвращает Uint8Array(48*48) серых пикселей (как PIL convert('L')). */
export function cropGray(video, box) {
  const { sx, sy, side } = squareRegion(box, video.videoWidth, video.videoHeight)
  ctx.drawImage(video, sx, sy, side, side, 0, 0, SIZE, SIZE)
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE)
  const gray = new Uint8Array(SIZE * SIZE)
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    gray[i] = (data[j] * 299 + data[j + 1] * 587 + data[j + 2] * 114) / 1000
  }
  return gray
}

/** PNG (data URL) из серого 48x48 — для сохранения своих снимков на бэкенде. */
export function grayToPng(gray) {
  const img = ctx.createImageData(SIZE, SIZE)
  for (let i = 0; i < gray.length; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = gray[i]
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return canvas.toDataURL('image/png')
}
