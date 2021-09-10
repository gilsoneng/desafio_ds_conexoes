# desafio_ds_conexoes
1 - Ter o Docker instalado no ambiente a qual será executado
2 - O comando -> "docker build -t desafio_ds ." constrói a imagem
3 - O comando -> "docker run desafio_ds:latest" executa a imagem
4 - Obter o Id do container a qual foi executado
	Neste caso pode-se utilizar o comando "docker ps -a -q" para listar somente os Container ID
	ou o comando "docker ps -a" para obter todas as informações
5 - 	Executar o seguinte código para copiar os arquivos com as classificações da imagem do Docker para a pasta do Host
	docker "cp [ContainerID]:/code/EVALUATED [/path/folder]"
	ex. "docker cp cb715b7194d3:/code/EVALUATED ~/desafio_ds_conexoes/"
