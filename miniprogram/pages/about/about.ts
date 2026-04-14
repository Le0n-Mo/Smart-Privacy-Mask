Page({
  data: {
    version: '1.0.0'
  },

  copyLink() {
    wx.setClipboardData({
      data: 'https://github.com/YourName/SmartRedact',
      success: () => wx.showToast({ title: '链接已复制' })
    })
  }
})