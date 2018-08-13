import { Component } from '@angular/core';
import { DogBreedService } from '../../services/dog-breed.service'
import {
    MatButtonModule,
    MatDialog,
    MatCardModule,
    MatProgressSpinnerModule
} from '@angular/material';


@Component({
    selector: 'dog-breed',
    templateUrl: './dog-breed.component.html',
    styleUrls: ['./dog-breed.component.css'],
    providers: [DogBreedService, MatDialog, MatCardModule, MatButtonModule, MatProgressSpinnerModule]
})

export class DogBreedComponent {
    title = 'Check the Dog breed';
    urlVariable: string;
    ourFile: File;
    isHidden = false;
    result: any = null;
    query_started: boolean = false;
    progress_spinner: boolean = false;

    constructor(private _service: DogBreedService) { }

    fileChange(files: File[]) {
        console.log('File change ')
        if (files.length > 0) {
            this.ourFile = files[0];
        }
    }


    openInput() {
        document.getElementById("fileInput").click();
    }


    upload() {
        console.log('sending this to server', this.ourFile);
        var reader = new FileReader();
        reader.readAsDataURL(this.ourFile);
        reader.onload = (event) => {
            this.urlVariable = reader.result;
            console.log('upload completed');
            this.isHidden = true;
            this.result = null;
            this.query_started = false;
            this.progress_spinner = false;
        }
    }

    submiImage() {
        this.query_started = true;
        this.progress_spinner = true;
        this.result = null;
        let formData: FormData = new FormData();
        formData.append('uploadFile', this.ourFile, this.ourFile.name);
        console.log('submitting image ', formData);
        this._service.getTheBreed(formData)
            .subscribe(result => {
                console.log('result ', result);
                this.result = result;
                this.result.result = result.result.split('.')[1].split('_').join(' ');
                this.progress_spinner = false;
            });
    }

}