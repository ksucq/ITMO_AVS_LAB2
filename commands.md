# лаба 2 по авс — команды узла б

делаем лабу вместе с артемом на двух макбуках через wi-fi

- узел а — ноут Артема, контейнер `data-processor`
- узел б — ноут Ксюши (мой), контейнер `ml-trainer`

работаем через терминал macos и docker desktop


## 1. подготовка

проверила docker:

```bash
docker --version
docker compose version
docker ps
docker ps -a
```

сначала docker не работал, пришлось открыть docker desktop

скачала проект с github:

```bash
git clone https://github.com/FreeGPTSubscriptions/ITMO_AVS_2_lab.git lab2
```

открыла проект в vs code и настроила свой контейнер:

```bash
cp node_b/.env.example node_b/.env
docker compose config
```

ошибок нету

## 2. проверка сети

узнала свой ip:

```bash
networksetup -listallhardwareports
ipconfig getifaddr en0
```

мой ip — `172.28.25.29`, артема — `172.28.25.218`

проверила соединение:

```bash
ping -c 4 172.28.25.218
route -n get 172.28.25.218
arp -n 172.28.25.218
```

у меня ping работал сразу, а у артема сначала была ошибка `no route to host`. после кучи проверок и мучений поняли, что дело в том, что артем вводил команды в терминале вс кода. попробовали подключение через обычный терминал macos, после этого соединение заработало и ноутбуки видели друг друга

## 3. настройка ssh

включила «удаленный вход» в настройках мака

проверила ssh:

```bash
sudo launchctl print system/com.openssh.sshd
```

создала файл `/etc/ssh/sshd_config.d/10-lab2.conf` и добавила:

```text
PermitRootLogin no
MaxAuthTries 3
```

проверила:

```bash
sudo /usr/sbin/sshd -t
sudo /usr/sbin/sshd -T | grep -E '^(permitrootlogin|maxauthtries) '
```

получилось `maxauthtries 3` и `permitrootlogin no`

артем создал ssh-ключ, а я добавила его публичный ключ в `~/.ssh/authorized_keys`

проверили подключение из контейнера артема:

```bash
docker exec data-processor ssh -i /run/ssh/lab2_ed25519 -o UserKnownHostsFile=/run/ssh/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=5 lugovaaksenia@172.28.25.29 whoami
```

получили `lugovaaksenia`. значит, подключение по ключу работает!!

## 4. запуск контейнера

запустила свой контейнер:

```bash
docker compose up --build -d
```

сначала не получилось, потому что порт 5000 был занят блин

проверила:

```bash
sudo lsof -nP -iTCP:5000 -sTCP:LISTEN
```

оказалось, что мешает airplay. отключила его в настройках, после чего контейнер запустился!

```bash
docker compose up --build -d
docker ps
curl http://127.0.0.1:5000/health
```

получила `{"data_ready":false,"status":"ok"}`. сервер уже работает, но файла от артема еще нет

## 5. передача csv и модель

артем запустил свой скрипт:

```bash
docker exec data-processor python /app/process_and_request.py
```

он очистил csv и передал его мне через scp

было 150 строк, удалилось 0

проверила, что файл появился у меня:

```bash
ls -lh ~/study/авс/lab2/node_b/data/clean_data.csv
docker exec ml-trainer ls -lh /home/DATA/clean_data.csv
curl http://127.0.0.1:5000/health
```

теперь сервер вернул `{"data_ready":true,"status":"ok"}`

проверила модель:

```bash
curl -s http://127.0.0.1:5000/predict | python3 -m json.tool
```

получила `status: success`. модель обучилась на 120 строках и проверилась на 30. точность вышла 93,33% (28 из 30 правильных ответов)

потом артем проверил запрос со своего ноутбука:

```bash
curl -s http://172.28.25.29:5000/predict | python3 -m json.tool
```

у него тоже все сработало, результат пришел с моего сервера

## 6. сетевой экран

по заданию нужно сделать так, чтобы к порту 5000 мог подключаться только артем

на маке вместо `iptables` есть `pf`. проверила его:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
sudo pfctl -s info
sudo pfctl -sr
sudo pfctl -s Anchors
```

оказалось, что `pf` уже включен и им пользуется мой red shield vpn

посмотрела его правила:

```bash
sudo pfctl -a com.redshieldvpn -sr
sudo pfctl -a 'com.redshieldvpn/*' -sr
sudo pfctl -a com.apple -sr
```

сохранила старые настройки:

```bash
sudo cp /etc/pf.conf /etc/pf.conf.lab2-backup
sudo pfctl -sr > ~/study/авс/lab2/pf_rules_before.txt
```

создала файл `lab2_pf.rules` и проверила его:

```bash
sudo pfctl -nf lab2_pf.rules
```

ошибок не было, но правила пока не включала, чтобы не сломать vpn

еще проверили сервер с третьего устройства через wi-fi. он тоже открылся, значит, порт 5000 пока доступен всем в нашей сети

этот пункт не доделали

## 7. конец 

для остановки контейнера:

```bash
docker stop ml-trainer
```
