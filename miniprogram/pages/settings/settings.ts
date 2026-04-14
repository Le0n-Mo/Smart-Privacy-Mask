Page({
  data: {
    config: {
      maskName: true,
      maskId: true,
      maskPhone: true,
      maskAddress: true
    }
  },

  onShow() {
    // 每次进入页面，从本地存储读取配置
    const savedConfig = wx.getStorageSync('mask_config');
    if (savedConfig) {
      this.setData({ config: savedConfig });
    }
  },

  updateConfig(e: any) {
    const key = e.currentTarget.dataset.key;
    const value = e.detail.value;
    const newConfig = { ...this.data.config, [key]: value };

    this.setData({ config: newConfig });
    
    // 同步到本地存储，实现“持久化”
    wx.setStorageSync('mask_config', newConfig);
    
    // 给用户一个反馈
    wx.showToast({ title: '配置已生效', icon: 'none', duration: 800 });
  }
})