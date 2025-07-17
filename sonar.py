import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import chirp, spectrogram
import streamlit as st
import os
from io import BytesIO, StringIO  # 用于在内存中处理文件

# 设置中文字体支持（确保部署后中文正常显示）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 全局配置参数
sample_rate = 1e6  # 1MHz
duration = 0.002   # 2ms
center_freq = 100e3  # 100kHz
sound_speed = 1500  # m/s
n = int(sample_rate * duration)
t = np.linspace(0, duration, n, endpoint=False)

# 磨蚀类型参数配置
abrasion_params = {
    "一般磨蚀": {
        "depth_range": (0.5, 6),
        "roughness": 0.5,
        "area_scale": 1.0,
        "crack_prob": 0.3,
        "region": "吸力面"
    },
    "点蚀": {
        "depth_range": (3, 6.5),
        "roughness": 0.8,
        "area_scale": 0.3,
        "crack_prob": 0.1,
        "region": "背水面"
    },
    "破损": {
        "depth_range": (8, 20),
        "roughness": 0.9,
        "area_scale": 1.5,
        "crack_prob": 0.9,
        "region": "根部"
    }
}

# 多回波信号生成函数（添加异常处理）
def generate_sonar_signal(abrasion_type, depth_scale=1.0, roughness_scale=1.0, n_echo=3):
    try:
        params = abrasion_params[abrasion_type]
        signal = np.zeros(n)

        pulse_length = int(0.0002 * sample_rate)
        pulse = np.sin(2 * np.pi * center_freq * t[:pulse_length]) * np.hanning(pulse_length)

        for i in range(n_echo):
            distance = np.random.uniform(0.1, 1.0)
            delay = int((2 * distance / sound_speed) * sample_rate)
            depth = np.random.uniform(*params["depth_range"]) * depth_scale
            amp = np.exp(-0.1 * depth) * (1 - 0.2 * i)
            echo = np.zeros(n)
            idx_end = min(n, delay + len(pulse))
            echo[delay:idx_end] = amp * pulse[:idx_end - delay]

            if params["region"] in ["吸力面", "背水面"]:
                phase_shift = np.random.normal(0, 0.05, idx_end - delay)
                echo[delay:idx_end] *= np.cos(phase_shift)

            signal += echo

        if abrasion_type == "点蚀":
            signal += np.sin(2 * np.pi * 500e3 * t) * np.random.normal(0, 0.1, n)
        elif abrasion_type == "破损":
            sweep = chirp(t, f0=50e3, f1=300e3, t1=duration, method='quadratic') * 0.1
            signal += sweep * np.random.normal(1.0, 0.2, n)

        signal += np.random.normal(0, 0.05 * params["roughness"] * roughness_scale, n)
        sand = np.random.uniform(0.5, 3.0)
        velocity = np.random.uniform(10, 30)
        signal += np.random.normal(0, 0.02 * sand + 0.03 * velocity/30, n)
        run_time = np.random.uniform(5, 30)
        signal *= (1 + run_time / 30 * 0.2)

        return t, signal, abrasion_type
    except Exception as e:
        st.error(f"信号生成失败：{str(e)}")
        return None, None, None

# 批量导出函数（适配云端，改为内存中生成文件供下载）
def export_dataset(selected_types, n_samples):
    try:
        # 存储所有文件的二进制数据，供用户下载
        files = []
        summary_data = []

        for abr in selected_types:
            for i in range(n_samples):
                t_vals, signal, abr_type = generate_sonar_signal(abr)
                if t_vals is None:
                    continue  # 跳过生成失败的样本

                # 生成CSV数据（内存中）
                df = pd.DataFrame({"时间(s)": t_vals, "信号幅度": signal, "磨蚀类别": abr_type})
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                csv_data = csv_buffer.getvalue().encode('utf-8')
                files.append((f"{abr}_{i+1}.csv", csv_data))
                summary_data.append(df)

                # 生成图像（内存中）
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(t_vals * 1000, signal)
                ax.set_xlabel("时间 (ms)")
                ax.set_ylabel("幅度")
                ax.set_title(f"{abr} 样本 {i+1}")
                fig.tight_layout()

                img_buffer = BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight')
                img_data = img_buffer.getvalue()
                files.append((f"{abr}_{i+1}.png", img_data))
                plt.close(fig)

        # 生成汇总CSV
        if summary_data:
            summary_df = pd.concat(summary_data)
            summary_buffer = StringIO()
            summary_df.to_csv(summary_buffer, index=False, encoding='utf-8-sig')
            summary_data = summary_buffer.getvalue().encode('utf-8')
            files.append(("信号汇总.csv", summary_data))

        return files
    except Exception as e:
        st.error(f"数据集生成失败：{str(e)}")
        return []

# Streamlit界面（优化交互和下载功能）
def gui():
    st.set_page_config(page_title="声纳信号生成器", layout="wide")  # 优化页面配置
    st.title("智能声纳信号生成器")
    st.markdown("无需安装任何软件，直接在浏览器中生成水斗磨蚀的声纳信号数据和图像。")

    # 信号生成区域
    with st.container(border=True):
        st.subheader("生成单个信号")
        col1, col2 = st.columns(2)
        with col1:
            abr_type = st.selectbox("选择磨蚀类型", list(abrasion_params.keys()))
            n_echo = st.slider("回波数量", 1, 5, 3)
        with col2:
            depth_scale = st.slider("磨蚀深度系数", 0.5, 2.0, 1.0)
            roughness_scale = st.slider("粗糙度系数", 0.5, 2.0, 1.0)

        generate_btn = st.button("生成信号", use_container_width=True)

        if generate_btn:
            t_vals, signal, abr = generate_sonar_signal(abr_type, depth_scale, roughness_scale, n_echo)
            if t_vals is None:
                st.stop()  # 生成失败时停止后续操作

            # 显示时域图
            st.subheader("时域图")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(t_vals * 1000, signal)
            ax1.set_xlabel("时间 (ms)")
            ax1.set_ylabel("信号幅度")
            ax1.set_title(f"{abr} 时域图")
            ax1.grid(True)
            st.pyplot(fig1)
            plt.close(fig1)

            # 显示频谱图
            st.subheader("频谱图")
            yf = fft(signal)
            xf = fftfreq(n, 1/sample_rate)[:n//2]
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(xf/1000, np.abs(yf[:n//2]))
            ax2.set_xlabel("频率 (kHz)")
            ax2.set_ylabel("幅度")
            ax2.set_title(f"{abr} 频谱图")
            ax2.grid(True)
            st.pyplot(fig2)
            plt.close(fig2)

            # 显示时频图
            st.subheader("时频图")
            f, t_spec, Sxx = spectrogram(signal, fs=sample_rate)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.pcolormesh(t_spec*1000, f/1000, 10*np.log10(Sxx), shading='gouraud')
            ax3.set_ylabel('频率 (kHz)')
            ax3.set_xlabel('时间 (ms)')
            ax3.set_title(f"{abr} 时频图")
            st.pyplot(fig3)
            plt.close(fig3)

            # 显示参数详情
            st.subheader(f"{abr} 参数详情")
            param_items = {k: str(v) for k, v in abrasion_params[abr_type].items()}
            st.dataframe(pd.DataFrame(param_items.items(), columns=['参数名称', '参数值']), use_container_width=True)

            # 单个文件下载按钮
            st.subheader("下载当前信号")
            df = pd.DataFrame({"时间(s)": t_vals, "信号幅度": signal, "磨蚀类别": abr})
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_data = csv_buffer.getvalue().encode('utf-8')
            st.download_button(
                label=f"下载 {abr} 信号 CSV",
                data=csv_data,
                file_name=f"sonar_{abr}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # 批量导出区域
    with st.container(border=True):
        st.subheader("批量导出数据集")
        selected_types = st.multiselect("选择要导出的磨蚀类型", list(abrasion_params.keys()))
        sample_num = st.number_input("每类样本数", min_value=1, max_value=20, value=3, step=1)  # 限制样本数，避免云端资源超限

        if st.button("生成并下载数据集", use_container_width=True):
            if not selected_types:
                st.warning("请至少选择一种磨蚀类型")
                st.stop()

            with st.spinner("正在生成数据集..."):
                files = export_dataset(selected_types, sample_num)

            if files:
                st.success(f"已生成 {len(files)} 个文件，可通过下方按钮下载")
                # 提供ZIP打包下载（需安装zipfile，云端通常已预装）
                import zipfile
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename, data in files:
                        zipf.writestr(filename, data)
                zip_buffer.seek(0)

                st.download_button(
                    label="下载完整数据集（ZIP）",
                    data=zip_buffer,
                    file_name="sonar_dataset.zip",
                    mime="application/zip",
                    use_container_width=True
                )

    # 添加使用说明
    with st.expander("使用说明"):
        st.markdown("""
        1. 生成单个信号：选择磨蚀类型和参数，点击"生成信号"，即可查看时域图、频谱图和时频图。
        2. 下载单个信号：生成信号后，可通过"下载当前信号"按钮获取CSV文件。
        3. 批量导出：选择需要的磨蚀类型和样本数，点击"生成并下载数据集"，获取包含CSV和图像的ZIP包。
        """)

# 主程序
if __name__ == "__main__":
    if hasattr(st, 'runtime') and st.runtime.exists():
        gui()
    else:
        print("请在Streamlit中运行此程序：")
        print("streamlit run sonar_signal_generator.py")