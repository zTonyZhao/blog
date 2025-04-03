---
title: 自建 tailscale DERP 中继服务器
categories:
- 开发
tags:
- Tailscale
- 云服务
index_img: /images/backgrounds/derp.svg
banner_img: /images/backgrounds/derp.svg
date: 2024-10-24 08:00:00
---

今年八月，我开启了在清深的研究生生活。

<!-- more -->

我在校外宿舍搭建了一台台式机，希望能够在上课时用平板访问这台主机，便于处理各种事务。

本科期间，北邮校园网限制较少，仅需在设备上搭建 DDNS 指向其 IPv6 地址即可非常方便的通过广域网进行访问。使用 Tailscale 更可免去 DDNS 的搭建，几乎所有场景下均可实现稳定的直连访问。

而来到深圳后，发现深圳大学城的网络是较为复杂的：其仅有 Portal 无线网，该无线网在 IPv6 层面仅支持有状态 DHCPv6，使得安卓设备无法正常获取 IPv6 地址，阻断了 IPv6 直连的可能性；其启用了 VLAN 隔离，致使内网无法直接互访；其防火墙较为严格，阻断了几乎全部的网外访问。这使得即便启用了 Tailscale 且两侧均为 IPv6 直连也会存在随机降级，链路不稳定的情况。

为了解决这一问题，遂决定搭建一台 DERP 中继服务器，在无法直连或直连不稳定时提供流量转发服务。

{% note danger %}
**重要提醒**
根据相关法律法规，此类服务仅限个人使用，不可用于盈利，**不可用于转发跨境流量**。
{% endnote %}

# 前期准备

为了搭建一个 tailscale DERP 服务器，你需要：

- 一台延迟较低的云服务器，其具备公网 IP 地址
- 一个已备案域名

在搭建 DERP 前，需要先行配置 DNS 条目，将域名指向服务器。

同时，需要放行两个自定义端口：一个 TCP 端口用于流量转发（DERP Port），一个 UDP 端口用于内网穿透（STUN Port）。

# 正式搭建

DERP 服务器需要一张有效的 SSL 证书，这里使用 acme.sh 进行申请。（也可跳过此步，由 derper 申请证书，详见下文）

我的服务器上已经安装了 nginx 并为域名编写了配置文件，故这里直接使用 acme.sh 的 nginx 模式，将证书部署在 `/etc/nginx/ssl.d/{domain}` 下。其他情况可参考 [官方Wiki](https://github.com/acmesh-official/acme.sh/wiki)。

```bash
curl https://get.acme.sh | sh -s email={Email}
acme.sh --set-default-ca --server letsencrypt
acme.sh --issue -d {domain} --nginx
mkdir -p /etc/nginx/ssl.d/{domain}
acme.sh --install-cert -d {domain} --key-file /etc/nginx/ssl.d/{domain}/{domain}.crt --fullchain-file /etc/nginx/ssl.d/{domain}/{domain}.key --reloadcmd "systemctl force-reload nginx"
```

获得证书后，在服务器上安装 golang，并获取官方的 DERP 服务器程序 `derper`。

需要注意的是，derper 没有自行升级的功能。在客户端版本更新后，官方会放出与客户端版本号相同的新版 derper，需要重新执行最后一行命令手动进行更新。

```bash
# 请将 golang 版本号替换成最新版本
wget https://go.dev/dl/go1.23.2.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.23.2.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
go env -w GOPROXY=https://goproxy.cn,direct
go install tailscale.com/cmd/derper@latest
```

derper 二进制可执行文件将被存放在 `~/go/bin/` 目录下，运行该文件即可启动中继服务器。

```bash
~/go/bin/derper -c ~/.derper.key -a :{DERP port} -http-port -1 -stun-port {STUN port} -hostname {domain} -certmode manual -certdir /etc/nginx/ssl.d/{domain}
```

其中，`-c` 指定服务器密钥的存放位置，`-a` 指定数据转发接口，`-stun-port` 指定打洞协议端口，`-http-port -1` 配置服务器仅使用 HTTPS 进行转发。

`-certmode` 默认为 `letsencrypt`，此时 derper 将会同时占用 80 端口自行申请 SSL 证书。这样配置比较省心，但端口使用效率较低，如非专用服务器不是非常推荐，推荐使用 `manual` 手动管理证书，将证书保存在 `{certdir}/{domain}.crt`, `{certdir}/{domain}.key` 中。

# 配置 derpMap

服务启动后，我们需要在管理台配置 derpMap 使中继服务器接入 tailnet。在 [ACL配置文件](https://login.tailscale.com/admin/acls/file) 的最后添加 derpMap 配置。

以下是一个示例配置文件。其中，RegionCode 建议使用 LOCODE（如深圳为 `snz`），与官方实现对齐。

```json
    "derpMap": {
		"Regions": {
			"900": {
				"RegionID": 900,
				"RegionCode": "{LOCODE}",
				"RegionName": "{region}",
				"Nodes": [
					{
						"Name": "900{nickname}",
						"RegionID": 900,
						"HostName": "{domain}",
						"IPv4": "{IPv4 address}",
						"IPv6": "{IPv6 address}",
						"DERPPort": {DERP port},
						"STUNPort": {STUN port},
					},
				],
			},
		},
	},
```

保存配置文件后，重启本地 Tailscale 客户端，命令行运行 `tailscale netcheck` 确认自建中继服务器工作情况。

![最快的 DERP 节点变为自建节点](netcheck.webp "验证自建中继服务器")

# 服务可持久化

刚刚运行的服务处于前台运行状态，按下 `ctrl+C` 或断开 SSH 连接后服务将会自行关闭。为了保持服务运行，需要编写 service 文件，将进程注册为服务。

创建 `/etc/derper`，将 derper 程序拷贝至该文件夹；创建 `/etc/derper/ssl.d`，将证书软链接至指定文件夹。

```bash
mkdir -p /etc/derper/ssl.d
cp ~/go/bin/derper /etc/derper/
ln -s /etc/nginx/ssl.d/{domain}/{domain}.crt /etc/derper/ssl.d/{domain}.crt
ln -s /etc/nginx/ssl.d/{domain}/{domain}.key /etc/derper/ssl.d/{domain}.key
```

编辑 `/etc/systemd/system/derper.service`，内容示例如下。

```ini
[Unit]
Description=Tailscale Derp Service
After=network.target

[Service]
ExecStart=/etc/derper/derper -c /etc/derper/derp.conf -a :{DERP port} -http-port -1 -stun-port {STUN port} -hostname {domain} -certmode manual -certdir /etc/derper/ssl.d
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

随后即可使用 `systemctl` 管理刚刚注册的服务。

```bash
systemctl daemon-reload
systemctl enable derper
systemctl start derper
```

# 访问控制

官方建议在服务器上同时安装 tailscale，通过配置将转发流量限制在 tailnet 内。

我不希望在云服务器上安装 tailscale，所以在这里使用另一种方法：通过 IPset 设定地区白名单的方式进行访问控制。具体内容详见下一篇文章。

附官方指南：https://tailscale.com/kb/1118/custom-derp-servers