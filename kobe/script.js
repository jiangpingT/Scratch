// 图片集交互功能
document.addEventListener('DOMContentLoaded', function() {
    // 创建模态框元素
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close-modal">&times;</span>
            <img class="modal-image" src="" alt="放大图片">
            <div class="modal-navigation">
                <button class="nav-btn prev-btn"><i class="fas fa-chevron-left"></i></button>
                <button class="nav-btn next-btn"><i class="fas fa-chevron-right"></i></button>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // 获取所有图片集中的图片
    const galleryItems = document.querySelectorAll('.gallery-item img');
    let currentIndex = 0;
    const modalImg = document.querySelector('.modal-image');
    const closeBtn = document.querySelector('.close-modal');
    const prevBtn = document.querySelector('.prev-btn');
    const nextBtn = document.querySelector('.next-btn');

    // 为每个图片添加点击事件
    galleryItems.forEach((img, index) => {
        // 添加懒加载属性
        img.setAttribute('loading', 'lazy');
        
        // 添加点击事件
        img.addEventListener('click', function() {
            modal.style.display = 'flex';
            modalImg.src = this.src;
            currentIndex = index;
            updateNavButtons();
        });

        // 预加载图片
        const preloadImg = new Image();
        preloadImg.src = img.src;
    });

    // 关闭模态框
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });

    // 点击模态框背景关闭
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // 上一张图片
    prevBtn.addEventListener('click', function() {
        currentIndex = (currentIndex - 1 + galleryItems.length) % galleryItems.length;
        modalImg.src = galleryItems[currentIndex].src;
        updateNavButtons();
    });

    // 下一张图片
    nextBtn.addEventListener('click', function() {
        currentIndex = (currentIndex + 1) % galleryItems.length;
        modalImg.src = galleryItems[currentIndex].src;
        updateNavButtons();
    });

    // 键盘导航
    document.addEventListener('keydown', function(e) {
        if (modal.style.display === 'flex') {
            if (e.key === 'ArrowLeft') {
                prevBtn.click();
            } else if (e.key === 'ArrowRight') {
                nextBtn.click();
            } else if (e.key === 'Escape') {
                modal.style.display = 'none';
            }
        }
    });

    // 更新导航按钮状态
    function updateNavButtons() {
        // 如果只有一张图片，隐藏导航按钮
        if (galleryItems.length <= 1) {
            prevBtn.style.display = 'none';
            nextBtn.style.display = 'none';
            return;
        }

        prevBtn.style.display = 'block';
        nextBtn.style.display = 'block';
    }

    // 添加图片加载动画效果
    galleryItems.forEach(img => {
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.5s ease-in-out';
        
        img.onload = function() {
            this.style.opacity = '1';
        };
        
        // 如果图片已经加载完成
        if (img.complete) {
            img.style.opacity = '1';
        }
    });
});

// 页面滚动动画
window.addEventListener('scroll', function() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    elements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        if (elementTop < window.innerHeight * 0.8) {
            element.classList.add('fade-in-up');
        }
    });
});

// 引用悬停效果
const quotes = document.querySelectorAll('blockquote');
quotes.forEach(quote => {
    quote.addEventListener('mouseenter', function() {
        this.style.transform = 'scale(1.02) rotate(1deg)';
        this.style.boxShadow = '0 8px 20px rgba(0,0,0,0.15)';
    });
    
    quote.addEventListener('mouseleave', function() {
        this.style.transform = 'scale(1) rotate(0)';
        this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
    });
});