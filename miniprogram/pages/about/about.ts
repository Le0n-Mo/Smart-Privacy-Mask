Page({
  data: {
    version: '1.0.0'
  },

  copyLink() {
    wx.setClipboardData({
      data: 'https://github.com/Le0n-Mo/Smart-Privacy-Mask',
      success: () => wx.showToast({ title: '链接已复制' })
    })
  }
})